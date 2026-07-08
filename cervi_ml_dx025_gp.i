# Cervi test1 tank, GP-corrected mixing-length case, dx=0.025
#
# Dimensional setup:
#   rho = 927 kg/m^3
#   mu  = 1.84e-5 Pa s
#   inlet velocity = (0, 0.05) m/s
#
# Standard ML:
#   l_m,std = min(kappa*y_w, 0.09*delta)
#   delta = 0.10 m = half inlet width
#
# GP correction:
#   xi = y_w / Lmax_eta
#   Lmax_eta = 0.4875 m, from the dx=0.025 MOOSE wall-distance max
#
#   factor = 1 + M_tank * A*tanh(gain*(xi-xi0)*xi)
#
# Tank mask:
#   active mainly in main tank:
#     y > 0, x < 1
#   inlet and outlet ducts remain close to standard ML.

rho = 927
mu = 1.84e-5

mesh = 'mesh/cervi_test1_dx025.msh'

advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

kappa_ml = 0.41
delta_ml = 0.10

Lmax_eta = 0.4875

activation_xi_gp = 0.0967
correction_amplitude_gp = 0.8
gp_gain = 26.7

y_mask_gp = 0.0
y_mask_width_gp = 0.05

x_mask_gp = 1.0
x_mask_width_gp = 0.05

inlet_u = 0.0
inlet_v = 0.05

[GlobalParams]
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = INSFVRhieChowInterpolator
    u = u
    v = v
    pressure = pressure
  []
[]

[Mesh]
  [mesh_file]
    type = FileMeshGenerator
    file = ${mesh}
  []
[]

[Problem]
  fv_bcs_integrity_check = false
[]

[Variables]
  [u]
    type = INSFVVelocityVariable
  []

  [v]
    type = INSFVVelocityVariable
  []

  [pressure]
    type = INSFVPressureVariable
  []
[]

[AuxVariables]
  [mixing_length_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [mixing_length_std_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [gp_factor_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [eddy_viscosity_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [elvol_aux_var]
    order  = CONSTANT
    family = MONOMIAL
  []

  [yw_aux_var]
    order  = CONSTANT
    family = MONOMIAL
  []
[]

[FVKernels]
  [mass]
    type = INSFVMassAdvection
    variable = pressure
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [u_time]
    type = INSFVMomentumTimeDerivative
    momentum_component = 'x'
    variable = u
    rho = ${rho}
  []

  [u_advection]
    type = INSFVMomentumAdvection
    momentum_component = 'x'
    variable = u
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [u_viscosity]
    type = INSFVMomentumDiffusion
    variable = u
    mu = ${mu}
    momentum_component = 'x'
  []

  [u_viscosity_rans]
    type = INSFVMixingLengthReynoldsStress
    variable = u
    rho = ${rho}
    mixing_length = mixing_length_aux_var
    momentum_component = 'x'
    u = u
    v = v
  []

  [u_pressure]
    type = INSFVMomentumPressure
    variable = u
    momentum_component = 'x'
    pressure = pressure
  []

  [v_time]
    type = INSFVMomentumTimeDerivative
    momentum_component = 'y'
    variable = v
    rho = ${rho}
  []

  [v_advection]
    type = INSFVMomentumAdvection
    momentum_component = 'y'
    variable = v
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [v_viscosity]
    type = INSFVMomentumDiffusion
    variable = v
    mu = ${mu}
    momentum_component = 'y'
  []

  [v_viscosity_rans]
    type = INSFVMixingLengthReynoldsStress
    variable = v
    rho = ${rho}
    mixing_length = mixing_length_aux_var
    momentum_component = 'y'
    u = u
    v = v
  []

  [v_pressure]
    type = INSFVMomentumPressure
    variable = v
    momentum_component = 'y'
    pressure = pressure
  []
[]

[AuxKernels]
  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'walls'
    variable = yw_aux_var
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  [mixing_len_std_aux_ker]
    type = WallDistanceMixingLengthAux
    walls = 'walls'
    variable = mixing_length_std_aux_var
    von_karman_const = ${kappa_ml}
    delta = ${delta_ml}
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  [gp_factor_aux_ker]
    type = ParsedAux
    variable = gp_factor_aux_var
    coupled_variables = 'yw_aux_var'
    use_xyzt = true
    expression = '1.0 + (0.5*(1.0 + tanh((y - ${y_mask_gp})/${y_mask_width_gp}))) * (0.5*(1.0 + tanh((${x_mask_gp} - x)/${x_mask_width_gp}))) * if(yw_aux_var/${Lmax_eta} < ${activation_xi_gp}, 0.0, ${correction_amplitude_gp}*tanh(${gp_gain}*(yw_aux_var/${Lmax_eta} - ${activation_xi_gp})*(yw_aux_var/${Lmax_eta})))'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  [mixing_len_aux_ker]
    type = ParsedAux
    variable = mixing_length_aux_var
    coupled_variables = 'mixing_length_std_aux_var gp_factor_aux_var'
    expression = 'mixing_length_std_aux_var * gp_factor_aux_var'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  [eddy_viscosity_aux_ker]
    type = INSFVMixingLengthTurbulentViscosityAux
    variable = eddy_viscosity_aux_var
    mixing_length = mixing_length_aux_var
    u = u
    v = v
    execute_on = 'TIMESTEP_END FINAL'
  []

  [elvol_aux_ker]
    type = VolumeAux
    variable = elvol_aux_var
    execute_on = 'INITIAL'
  []
[]

[ICs]
  [u_ic]
    type = ConstantIC
    variable = u
    value = 0
  []

  [v_ic]
    type = ConstantIC
    variable = v
    value = 0
  []

  [p_ic]
    type = ConstantIC
    variable = pressure
    value = 0
  []
[]

[FVBCs]
  [inlet-u]
    type     = INSFVInletVelocityBC
    boundary = 'inlet'
    variable = u
    functor  = ${inlet_u}
  []

  [inlet-v]
    type     = INSFVInletVelocityBC
    boundary = 'inlet'
    variable = v
    functor  = ${inlet_v}
  []

  [walls-u]
    type     = INSFVNoSlipWallBC
    boundary = 'walls'
    variable = u
    function = 0
  []

  [walls-v]
    type     = INSFVNoSlipWallBC
    boundary = 'walls'
    variable = v
    function = 0
  []

  [outlet-p]
    type     = INSFVOutletPressureBC
    boundary = 'outlet'
    variable = pressure
    function = 0
  []
[]

[VectorPostprocessors]
  [vpp]
    type = ElementValueSampler
    variable = 'u v pressure mixing_length_aux_var mixing_length_std_aux_var gp_factor_aux_var eddy_viscosity_aux_var'
    sort_by = x
  []

  [elv]
    type = ElementValueSampler
    variable = 'elvol_aux_var yw_aux_var mixing_length_aux_var mixing_length_std_aux_var gp_factor_aux_var'
    sort_by = x
    execute_on = 'INITIAL'
  []
[]

[Preconditioning]
  [SMP_PJFNK]
    type = SMP
    full = true
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type -pc_factor_mat_solver_type -ksp_gmres_restart'
    petsc_options_value = 'lu superlu_dist 100'
  []
[]

[Executioner]
  type = Transient

  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.25
    optimal_iterations = 8
    linear_iteration_ratio = 150
    dt = 1e-4
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.75
  []

  dtmin = 1e-10
  dtmax = 50

  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
  nl_max_its = 50

  l_tol = 1e-5
  l_max_its = 100

  start_time = 0
  end_time  = 5000
  num_steps = 10000

  steady_state_detection = true
  steady_state_tolerance = 1e-6
[]

[Outputs]
  print_linear_residuals = false

  [exodus]
    type = Exodus
    execute_on = FINAL
    file_base = cervi_ml_dx025_gp
  []

  [csv]
    type = CSV
    execute_on = FINAL
    file_base = cervi_ml_dx025_gp
  []

  [out]
    type = Checkpoint
    execute_on = FINAL
    file_base = cervi_ml_dx025_gp_cp
  []
[]
