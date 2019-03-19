#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "simplelevel_feature_extractor.h"
#include <rcsc/common/server_param.h>

using namespace rcsc;

SimpleLevelFeatureExtractor::SimpleLevelFeatureExtractor(int num_teammates,
                                                   int num_opponents,
                                                   bool playing_offense) :
    FeatureExtractor(num_teammates, num_opponents, playing_offense)
{
  assert(numTeammates >= 0);
  assert(numOpponents >= 0);
  numFeatures = num_basic_features + features_per_player * (numTeammates + numOpponents);
  feature_vec.resize(numFeatures);
}

SimpleLevelFeatureExtractor::~SimpleLevelFeatureExtractor() {}

const std::vector<float>&
SimpleLevelFeatureExtractor::ExtractFeatures(const rcsc::WorldModel& wm, 
            bool last_action_status, hfo::Player player_on_ball, long ep_end_time) {
  featIndx = 0;
  const ServerParam& SP = ServerParam::i();
  // ======================== SELF FEATURES ======================== //
  const SelfObject& self = wm.self();
  const Vector2D& self_pos = self.pos();
  const AngleDeg& self_ang = self.body();
  const PlayerPtrCont& teammates = wm.teammatesFromSelf();
  const PlayerPtrCont& opponents = wm.opponentsFromSelf();

  const BallObject& ball = wm.ball();
  addFeature(ball.pos().x/pitchHalfLength);
  addFeature(ball.pos().y/pitchHalfWidth);
  addFeature(ball.vel().x/pitchHalfLength);
  addFeature(ball.vel().y/pitchHalfWidth);

  // Self features
  addFeature(self_pos.x/pitchHalfLength);
  addFeature(self_pos.y/pitchHalfWidth);
  addFeature(self.vel().x/pitchHalfLength);
  addFeature(self.vel().y/pitchHalfWidth);
  addAngFeature(self_ang);
  addNormFeature(self.stamina(), 0, SP.staminaMax());

  // episode time step
  addFeature((wm.fullstateTime().cycle() - ep_end_time)/1000.0);

  assert(featIndx == num_basic_features);

  // teammate's x, y, vel_x, vel_y, sin_ang, cos_ang
  int detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it=teammates.begin(); it != teammates.end(); ++it) {
    const PlayerObject* teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates) {
      addFeature(teammate->pos().x/pitchHalfLength);
      addFeature(teammate->pos().y/pitchHalfWidth);
      addFeature(teammate->vel().x/pitchHalfLength);
      addFeature(teammate->vel().y/pitchHalfWidth);
      addAngFeature(teammate->body());
      detected_teammates++;
    }
  }
  // Add -2 features for any missing teammates
  for (int i=detected_teammates; i<numTeammates; ++i) {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  // opponent's x, y, vel_x, vel_y, sin_ang, cos_ang
  int detected_opponents = 0;
  for (PlayerPtrCont::const_iterator it=opponents.begin(); it != opponents.end(); ++it) {
    const PlayerObject* opponent = *it;
    if (valid(opponent) && opponent->unum() > 0 && detected_opponents < numOpponents) {
      addFeature(opponent->pos().x/pitchHalfLength);
      addFeature(opponent->pos().y/pitchHalfWidth);
      addFeature(opponent->vel().x/pitchHalfLength);
      addFeature(opponent->vel().y/pitchHalfWidth);
      addAngFeature(opponent->body());
      detected_opponents++;
    }
  }
  // Add -2 features for any missing teammates
  for (int i=detected_opponents; i<numOpponents; ++i) {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  assert(featIndx == numFeatures);
  checkFeatures();
  return feature_vec;
}

bool SimpleLevelFeatureExtractor::valid(const rcsc::PlayerObject* player) {
  if (!player) {return false;} //avoid segfaults
  const rcsc::Vector2D& pos = player->pos();
  if (!player->posValid()) {
    return false;
  }
  return pos.isValid();
}

// void SimpleLevelFeatureExtractor::addLandmarks(const rcsc::SideID side) {
//   if(side == rcsc::LEFT) {
//     // self top corner x
//     addFeature(-pitchHalfLength/pitchHalfLength);
//     // self top corner y
//     addFeature(-pitchHalfWidth/pitchHalfWidth);
//     //
//   }
// }