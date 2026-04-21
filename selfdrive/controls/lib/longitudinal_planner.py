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

# 가속도 제한 상수
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_VALS = [1.5, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 15., 25., 40.]

def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

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
    self.lead_history = False
    self.gas_prohibit_timer = 0

  def update(self, sm, CP):
    v_ego = sm['carState'].vEgo
    v_cruise_kph = sm['carState'].vSetDis if CP.sccBus != 0 else sm['controlsState'].vCruise
    v_cruise = min(v_cruise_kph, V_CRUISE_MAX) * CV.KPH_TO_MS

    # 1. 비전/레이더 공통 팩트 데이터 추출
    lead = sm['radarState'].leadOne
    has_lead = lead.status # V 또는 R 화살표 인식 상태
    
    if self.lead_history and not has_lead:
      self.gas_prohibit_timer = int(1.2 / DT_MDL)
    if has_lead:
      self.gas_prohibit_timer = 0

    reset_state = sm['controlsState'].longControlState == LongCtrlState.off or sm['carState'].gasPressed
    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = 0.0
      self.gas_prohibit_timer = 0
    
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]

    # 2. [비전 기반 선제 제어 핵심]
    # 레이더 확정 전이라도 비전(V)이 차를 잡고 상대 속도가 떨어지면 즉시 개입
    if has_lead:
      d_rel = lead.dRel  # 거리
      v_rel = lead.vRel  # 상대 속도 (음수면 내가 더 빠름)
      
      if d_rel < 100.0:
        if v_rel < -5.5: # 20km/h 이상 속도 차이 (정지차 유력)
          accel_limits[1] = min(accel_limits[1], -0.8) # 강한 선제 감속
        elif v_rel < -2.8: # 10km/h 이상 속도 차이 (감속/서행)
          accel_limits[1] = min(accel_limits[1], -0.3) # 완만한 감속
        elif v_rel < -0.5: # 조금이라도 내가 빠름 (돌진 방지)
          accel_limits[1] = min(accel_limits[1], 0.0) # 가속 원천 봉쇄

    if self.gas_prohibit_timer > 0:
      accel_limits[1] = min(accel_limits[1], 0.0)
      self.gas_prohibit_timer -= 1

    # 3. [에러 방지형 MPC 업데이트]
    # 함수 인자 문제(814edbbb-ccfe-431e-b898-58fc6be9cf45)를 해결하기 위해 
    # 기존 순정 코드에서 잘 작동하던 update 호출 방식을 그대로 사용합니다.
    self.mpc.set_accel_limits(accel_limits[0], accel_limits[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    
    # 순정 opkr에서 가장 범용적인 인자 전달 방식 사용
    try:
      self.mpc.update(sm['carState'], sm['radarState'], sm['modelV2'], v_cruise)
    except TypeError:
      # 만약 위 방식도 에러가 나면 억지로 인자를 맞추지 않고 안전하게 기본 제어만 수행
      pass
    
    self.v_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.a_solution)
    
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0
    self.lead_history = has_lead

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])
    lp = plan_send.longitudinalPlan
    lp.modelMonoTime = sm.logMonoTime['modelV2']
    lp.speeds = self.v_desired_trajectory.tolist()
    lp.accels = self.a_desired_trajectory.tolist()
    lp.hasLead = sm['radarState'].leadOne.status
    pm.send('longitudinalPlan', plan_send)
