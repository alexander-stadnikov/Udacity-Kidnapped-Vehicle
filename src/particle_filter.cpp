/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
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

namespace {
bool is_zero(double val)
{
  return std::isless(val, 1.0e-06);
}
}

void ParticleFilter::init(double x, double y, double theta, double sigma[])
{
  num_particles = 1000;
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0);

  // Use Gaussian distribution
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, sigma[0]);
  std::normal_distribution<double> dist_y(y, sigma[1]);
  std::normal_distribution<double> dist_theta(theta, sigma[2]);

  for (size_t i = 0; i < particles.size(); i++) {
    auto& p = particles[i];
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

  const auto dTheta = yaw_rate * delta_t;
  const double vPerTheta = is_zero(yaw_rate) ? 0.0 : velocity / yaw_rate;
  const auto dV = velocity * delta_t;

  for (auto& p : particles) {
    const auto cosP = cos(p.theta);
    const auto sinP = sin(p.theta);
    if (is_zero(yaw_rate)) {
      p.x += dV * cosP;
      p.y += dV * sinP;
    } else {
      const auto theta_2 = p.theta + dTheta;
      p.x += vPerTheta * (sin(theta_2) - sinP);
      p.y += vPerTheta * (cosP - cos(theta_2));
      p.theta += dTheta;
    }

    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs>& observations)
{
  const auto not_found = -1;
  for (auto& o : observations) {
    auto min_dist = std::numeric_limits<double>::max();
    o.id = not_found;
    for (const auto& p : predicted) {
      const auto d = dist(p.x, p.y, o.x, o.y);      
      if (std::islessequal(d, min_dist)) {
        min_dist = d;
        o.id = p.id;
      }
    }
    
    if (o.id == not_found) {
      throw std::logic_error("Landmark not found");
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, 
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}