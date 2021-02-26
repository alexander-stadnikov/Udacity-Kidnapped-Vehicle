/**
 * particle_filter.cpp
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "helper_functions.h"

namespace
{
  std::vector<LandmarkObs> observations_for_particle(const Particle &p,
                                                     const std::vector<LandmarkObs> &obs)
  {
    const double cosTheta = cos(p.theta);
    const double sinTheta = sin(p.theta);

    std::vector<LandmarkObs> out;
    out.reserve(obs.size());
    for (const auto &o : obs)
    {
      out.push_back({.id = -1,
                     .x = p.x + cosTheta * o.x - sinTheta * o.y,
                     .y = p.y + sinTheta * o.x + cosTheta * o.y});
    }

    return out;
  }

  std::vector<LandmarkObs> filter_landmarks(double sensor_range, const Particle &p,
                                            const std::vector<Map::single_landmark_s> &landmarks)
  {
    std::vector<LandmarkObs> out;
    out.reserve(landmarks.size());
    for (auto const &l : landmarks)
    {
      if (std::islessequal(dist(p.x, p.y, l.x_f, l.y_f), sensor_range))
      {
        out.push_back({
            .id = l.id_i,
            .x = static_cast<double>(l.x_f),
            .y = static_cast<double>(l.y_f),
        });
      }
    }

    return out;
  }

}

void ParticleFilter::init(double x, double y, double theta, double sigma[])
{
  num_particles = 500;
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0);

  // Use Gaussian distribution
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, sigma[0]);
  std::normal_distribution<double> dist_y(y, sigma[1]);
  std::normal_distribution<double> dist_theta(theta, sigma[2]);

  for (size_t i = 0; i < particles.size(); i++)
  {
    auto &p = particles[i];
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = weights[i];
    p.id = i;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // Use Gaussian distribution
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0.0, std_pos[0]);
  std::normal_distribution<double> noise_y(0.0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0.0, std_pos[2]);

  const auto yawRateIsSmall = (fabs(yaw_rate) < 1e-06);
  const auto dV = velocity * delta_t;

  for (auto &p : particles)
  {
    const auto cosP = cos(p.theta);
    const auto sinP = sin(p.theta);

    if (yawRateIsSmall)
    {
      p.x += dV * cosP;
      p.y += dV * sinP;
    }
    else
    {
      const auto dTheta = yaw_rate * delta_t;
      const double vPerTheta = velocity / yaw_rate;

      p.x += vPerTheta * (sin(p.theta + dTheta) - sinP);
      p.y += vPerTheta * (cosP - cos(p.theta + dTheta));
      p.theta += dTheta;
    }

    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations)
{
  const auto not_found = -1;
  for (auto &o : observations)
  {
    auto min_dist = std::numeric_limits<double>::max();
    o.id = not_found;
    for (const auto &p : predicted)
    {
      const auto d = dist(p.x, p.y, o.x, o.y);
      if (std::islessequal(d, min_dist))
      {
        min_dist = d;
        o.id = p.id;
      }
    }

    if (o.id == not_found)
    {
      throw std::logic_error("Landmark not found");
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  for (size_t pId = 0; pId < particles.size(); pId++)
  {
    auto &p = particles[pId];
    auto transformedObservations = observations_for_particle(p, observations);
    const auto &mapLandmarks = map_landmarks.landmark_list;
    const auto landmarks = filter_landmarks(sensor_range, p, mapLandmarks);

    dataAssociation(landmarks, transformedObservations);

    p.weight = 1.0;
    for (const auto &tobs : transformedObservations)
    {
      const double x = static_cast<double>(mapLandmarks[tobs.id - 1].x_f);
      const double y = static_cast<double>(mapLandmarks[tobs.id - 1].y_f);
      const double xDiff = pow(tobs.x - x, 2.0);
      const double yDiff = pow(tobs.y - y, 2.0);
      const double xStd = pow(std_landmark[0], 2.0);
      const double yStd = pow(std_landmark[1], 2.0);
      const double prob = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(xDiff / (2 * xStd) + yDiff / (2 * yStd)));
      p.weight *= prob;
    }

    weights[pId] = p.weight;
  }
}

void ParticleFilter::resample()
{
  std::default_random_engine gen;
  std::discrete_distribution<size_t> d(weights.begin(), weights.end());
  std::vector<Particle> resampled(particles.size());
  for (auto &p : resampled)
  {
    p = particles[d(gen)];
  }
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const std::vector<int> &associations,
                                     const std::vector<double> &sense_x,
                                     const std::vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord)
{
  std::vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
