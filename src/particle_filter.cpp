/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

//create static random generator
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(is_initialized){
		return;
	}
	
	num_particles = 100;
	//Add random Gaussian noise  
	
	normal_distribution<double> nd_x(x, std[0]);
	normal_distribution<double> nd_y(y, std[1]);
	normal_distribution<double> nd_theta(theta, std[2]);
	
	for(int i = 0; i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = nd_x(gen);
		particle.y = nd_y(gen);
		particle.theta = nd_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
//		weights.push_back(particle.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	//create noise generator
	normal_distribution<double> nd_x(0, std_pos[0]);
	normal_distribution<double> nd_y(0, std_pos[1]);
	normal_distribution<double> nd_theta(0, std_pos[2]);
		
	for(int i=0; i < num_particles; i++){
		double theta = particles[i].theta;
		if(fabs(yaw_rate) < EPS){
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}else{
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += velocity / yaw_rate * (-cos(theta + yaw_rate * delta_t) + cos(theta));
			particles[i].theta += yaw_rate * delta_t;
		}
		//add noise 
		particles[i].x += nd_x(gen);
		particles[i].y += nd_y(gen);
		particles[i].theta += nd_theta(gen);
	}
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++ ){
		double min_dist = numeric_limits<double>::max();
		int id = -1;
		for (int j = 0; j < predicted.size(); j++){
			double dist = pow(observations[i].x - predicted[j].x, 2) +
					pow(observations[i].y - predicted[j].y, 2);
			if(min_dist > dist){
				min_dist = dist;
				id = predicted[j].id;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		
		//Filter map landmarks
		vector<LandmarkObs> filteredLandmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
			if(fabs(p_x -  landmark.x_f) <= sensor_range 
					&& fabs(p_y - landmark.y_f) <= sensor_range){
				filteredLandmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}
		
		//Transform observation coordinates
		vector<LandmarkObs> transformedObservations;
		for(int j = 0; j < observations.size(); j++){
			double o_x = observations[j].x;
			double o_y = observations[j].y;
			double tx = cos(p_theta) * o_x - sin(p_theta) * o_y + p_x;
			double ty = sin(p_theta) * o_x + cos(p_theta) * o_y + p_y;
			transformedObservations.push_back(LandmarkObs{observations[j].id, tx, ty});
		}
		
		//Make date association
		dataAssociation(filteredLandmarks, transformedObservations);
		
		//reinit weight of particles
		particles[i].weight = 1.0;
		
		//new weight
		for (int j = 0; j < transformedObservations.size(); j++){
			double o_x = transformedObservations[j].x;
			double o_y = transformedObservations[j].y;
			double o_id = transformedObservations[j].id;
			bool found = false;
			int k = 0;
			double found_x, found_y;
			while(!found && k < filteredLandmarks.size()){
				double l_x = filteredLandmarks[k].x;
				double l_y = filteredLandmarks[k].y;
				double l_id = filteredLandmarks[k].id;
				if(o_id == l_id){
					found = true;
					found_x = filteredLandmarks[k].x;
					found_y = filteredLandmarks[k].y;
				}
				k++;
			}
			double delta_x = o_x - found_x;
			double delta_y = o_y - found_y;
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double weight = (1.0 / 2.0 / M_PI / s_x / s_y) * 
					exp(- pow(delta_x, 2) / 2.0 / pow(s_x, 2) - pow(delta_y, 2) / 2.0 / pow(s_y, 2));
			particles[i].weight *= weight;	
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	vector<double> weights;
	for(int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	int index = uniintdist(gen);
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	double beta = 0.0;
	for(int i = 0; i < num_particles; i++){
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	
	particles = new_particles;
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
