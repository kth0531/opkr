#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import interp

import cereal.messaging as messaging
from common.conversions import Conversions as CV
from common.filter_simple import FirstOrderFilter
from common.realtime import DT_MDL
from selfdrive.modeld.constants import T_IDXS
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N
from selfdrive.swaglog import cloudlog

LON_MPC_STEP = 0.2  # first step is 0.2s
AWARENESS_DECEL = -0.2  # car smoothly decel at .2m/s^2 when user is distracted
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_VALS = [1.5, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 15., 25., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class Planner:
  def __init__(self, CP, init_v=0.0, init_a=0.0):
    self.CP = CP
    self.mpc = LongitudinalMpc()

    self.fcw = False

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, DT_MDL)

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0

  def update(self, sm, CP):
    v_ego = sm['carState'].vEgo

    if CP.sccBus != 0:
      v_cruise_kph = sm['carState'].vSetDis
    else:
      v_cruise_kph = sm['controlsState'].vCruise
    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
# --- 수정된 통합 비전 기반 감속 로직 시작 ---
    if len(sm['modelV2'].leadsV3) > 0:
        lead_v3 = sm['modelV2'].leadsV3[0]
        v_ego = sm['carState'].vEgo
        v_ego_kph = v_ego * 3.6

        # 1. 비전 확률 60% 이상 시 작동
        if lead_v3.prob > 0.6:
            d_rel = lead_v3.x[0]
            v_rel = lead_v3.v[0]

            # [상황 A] 시속 50km/h 이상 고속 주행 시
            if v_ego_kph >= 50:
                # 초장거리 대응 (120m~250m) & 상대속도 18km/h 이상 차이 시
                if 120 < d_rel <= 250 and v_rel < -5.0:
                    v_cruise_kph = max(v_cruise_kph - 5, 30)
                
                # 장거리 적극 대응 (100m 이내) & 상대속도 12km/h 이상 차이 시
                elif d_rel <= 100 and v_rel < -(12.0 / 3.6):
                    v_cruise_kph = max(v_cruise_kph - 15, 30)

            # [상황 B] 단거리 정지차 대응 (확률 70% 이상, 40m 이내)
            if lead_v3.prob > 0.7 and d_rel < 40 and v_rel < -5.0:
                v_cruise_kph = max(v_cruise_kph - 20, 30)

    # --- 통합 로직 끝 ---
# --- 초록불 감지 시 출발 지원 로직 (실험적) ---
    # 차가 정지 상태(v_ego < 0.1)일 때만 작동
    if v_ego < 0.1 and len(sm['modelV2'].trafficLights) > 0:
        tl = sm['modelV2'].trafficLights[0]  # 가장 확률 높은 신호등 데이터
        
        # 1. 초록불일 확률이 90% 이상이고
        # 2. 빨간불 확률이 10% 미만이며
        # 3. 좌회전 화살표 확률이 10% 미만일 때 (좌회전 신호 필터링)
        if tl.state == 3 and tl.prob > 0.9:
            # 브랜치 모델에 따라 redProb, leftArrowProb 변수명이 다를 수 있음
            # 일반적인 콤마 모델 기준의 필터링 로직
            is_pure_green = True 
            
            # 좌회전 신호(빨간불+화살표)를 거르기 위한 이중 체크
            if hasattr(tl, 'redProb') and tl.redProb > 0.1:
                is_pure_green = False
            if hasattr(tl, 'leftArrowProb') and tl.leftArrowProb > 0.1:
                is_pure_green = False

            if is_pure_green:
                # [액션 1] 크루즈 설정 속도를 현재 속도보다 높게 설정하여 출발 대기
                v_cruise_kph = max(v_cruise_kph, 30)
                
                # [액션 2] 앞차와의 가상 거리를 벌려 SCC가 출발하도록 유도 (매우 실험적)
                # 실제 물리적 출발은 현대차 SCC 특성상 RES 버튼이나 가속 페달이 필요할 수 있음
                # 우선 v_cruise를 띄워 '준비' 상태로 만듭니다.
                v_cruise = v_cruise_kph * CV.KPH_TO_MS
    # --- 초록불 로직 끝 ---

    
    long_control_state = sm['controlsState'].longControlState
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_state == LongCtrlState.off
    reset_state = reset_state or sm['carState'].gasPressed

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = 0.0

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]
    accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    if force_slow_decel and False: # awareness decel is disabled for now:
      # if required so, force a smooth deceleration
      accel_limits_turns[1] = min(accel_limits_turns[1], AWARENESS_DECEL)
      accel_limits_turns[0] = min(accel_limits_turns[0], accel_limits_turns[1])
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.05)
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)

    self.mpc.set_weights(prev_accel_constraint)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    if (len(sm['modelV2'].position.x) == 33 and
         len(sm['modelV2'].velocity.x) == 33 and
          len(sm['modelV2'].acceleration.x) == 33):
      x = np.interp(T_IDXS_MPC, T_IDXS, sm['modelV2'].position.x)
      v = np.interp(T_IDXS_MPC, T_IDXS, sm['modelV2'].velocity.x)
      a = np.interp(T_IDXS_MPC, T_IDXS, sm['modelV2'].acceleration.x)
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
    self.mpc.update(sm['carState'], sm['radarState'], sm['modelV2'], v_cruise, x, v, a)
    self.v_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 5 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.dynamicTRMode = int(self.mpc.dynamic_TR_mode)
    longitudinalPlan.dynamicTRValue = float(self.mpc.desired_TR)

    longitudinalPlan.e2eX = self.mpc.e2e_x.tolist()
    longitudinalPlan.lead0Obstacle = self.mpc.lead_0_obstacle.tolist()
    longitudinalPlan.lead1Obstacle = self.mpc.lead_1_obstacle.tolist()
    longitudinalPlan.cruiseTarget = self.mpc.cruise_target.tolist()
    longitudinalPlan.stopLine = self.mpc.stopline.tolist()
    longitudinalPlan.stoplineProb = self.mpc.stop_prob

    pm.send('longitudinalPlan', plan_send)
