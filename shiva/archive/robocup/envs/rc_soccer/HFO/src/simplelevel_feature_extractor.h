// -*-c++-*-

#ifndef SIMPLELEVEL_FEATURE_EXTRACTOR_H
#define SIMPLELEVEL_FEATURE_EXTRACTOR_H

#include <rcsc/player/player_agent.h>
#include "feature_extractor.h"
#include "common.hpp"
#include <vector>

class SimpleLevelFeatureExtractor : public FeatureExtractor {
public:
  SimpleLevelFeatureExtractor(int num_teammates, int num_opponents,
                           bool playing_offense);
  virtual ~SimpleLevelFeatureExtractor();

  // Updated the state features stored in feature_vec
  virtual const std::vector<float>& ExtractFeatures(const rcsc::WorldModel& wm,
            bool last_action_status, hfo::Player player_on_ball, long ep_end_time);
  
  //override FeatureExtractor::valid
  //this method takes a pointer instead of a reference
  bool valid(const rcsc::PlayerObject* player);

protected:
  // Number of unvaried indeces
  const static int num_basic_features = 16 + 12; // landmarks (16), ball, self, time step
  // Number of features for each teammate or opponent in game
  const static int features_per_player = 6;

// private:
    // void addLandmarks(const rcsc::SideID side);
  
};

#endif // SIMPLELEVEL_FEATURE_EXTRACTOR_H