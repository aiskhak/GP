# Non-dimensionalization:
#   U_avg,inlet = 1
#   D_inlet     = 1
#   rho         = 1
#   Re_D        = 4000
#   mu          = 1/Re_D

mu  = 2.5e-4
rho = 1

mesh = '../../mesh/migadome.msh'

mesh_scale = 5.714285714285714e-2  # 1/17.5

advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

# Standard mixing-length parameters
kappa_ml = 0.41
delta_ml = 0.5       # because scaled inlet diameter = 1, radius = 0.5

[GlobalParams]
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = INSFVRhieChowInterpolator
    u = u
    v = v
    w = w
    pressure = pressure
  []
[]

[Mesh]
  [mesh_file]
    type = FileMeshGenerator
    file = ${mesh}
  []

  [scale]
    type = TransformGenerator
    input = mesh_file
    transform = SCALE
    vector_value = '${mesh_scale} ${mesh_scale} ${mesh_scale}'
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

  [w]
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

[Functions]
  [v_in]
    type = ParsedFunction
    expression = 'if(t < 1.0, t/1.0, 1.0)*if(4*(x^2 + (z - 2.42857142857143)^2) < 1, 1.3333333333333333*(1 - (4*(x^2 + (z - 2.42857142857143)^2))^3), 0)'
  []

[v_start]
  type = ParsedFunction
  expression = '1e-4 + 1e-5*y'
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

  # x-momentum
  [u_time]
    type = INSFVMomentumTimeDerivative
    variable = u
    momentum_component = 'x'
    rho = ${rho}
  []

  [u_advection]
    type = INSFVMomentumAdvection
    variable = u
    momentum_component = 'x'
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [u_viscosity]
    type = INSFVMomentumDiffusion
    variable = u
    momentum_component = 'x'
    mu = ${mu}
  []

  [u_viscosity_rans]
    type = INSFVMixingLengthReynoldsStress
    variable = u
    momentum_component = 'x'
    rho = ${rho}
    mixing_length = mixing_length_aux_var
    u = u
    v = v
    w = w
  []

  [u_pressure]
    type = INSFVMomentumPressure
    variable = u
    momentum_component = 'x'
    pressure = pressure
  []

  # y-momentum
  [v_time]
    type = INSFVMomentumTimeDerivative
    variable = v
    momentum_component = 'y'
    rho = ${rho}
  []

  [v_advection]
    type = INSFVMomentumAdvection
    variable = v
    momentum_component = 'y'
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [v_viscosity]
    type = INSFVMomentumDiffusion
    variable = v
    momentum_component = 'y'
    mu = ${mu}
  []

  [v_viscosity_rans]
    type = INSFVMixingLengthReynoldsStress
    variable = v
    momentum_component = 'y'
    rho = ${rho}
    mixing_length = mixing_length_aux_var
    u = u
    v = v
    w = w
  []

  [v_pressure]
    type = INSFVMomentumPressure
    variable = v
    momentum_component = 'y'
    pressure = pressure
  []

  # z-momentum
  [w_time]
    type = INSFVMomentumTimeDerivative
    variable = w
    momentum_component = 'z'
    rho = ${rho}
  []

  [w_advection]
    type = INSFVMomentumAdvection
    variable = w
    momentum_component = 'z'
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [w_viscosity]
    type = INSFVMomentumDiffusion
    variable = w
    momentum_component = 'z'
    mu = ${mu}
  []

  [w_viscosity_rans]
    type = INSFVMixingLengthReynoldsStress
    variable = w
    momentum_component = 'z'
    rho = ${rho}
    mixing_length = mixing_length_aux_var
    u = u
    v = v
    w = w
  []

  [w_pressure]
    type = INSFVMomentumPressure
    variable = w
    momentum_component = 'z'
    pressure = pressure
  []
[]

[AuxKernels]
  [mixing_len_aux_ker]
    type = WallDistanceMixingLengthAux
    walls = 'Wall'
    variable = mixing_length_aux_var
    von_karman_const = ${kappa_ml}
    delta = ${delta_ml}
    execute_on = 'INITIAL'
  []

  [eddy_viscosity_aux_ker]
    type = INSFVMixingLengthTurbulentViscosityAux
    variable = eddy_viscosity_aux_var
    mixing_length = mixing_length_aux_var
    u = u
    v = v
    w = w
    execute_on = 'TIMESTEP_END FINAL'
  []

  [elvol_aux_ker]
    type = VolumeAux
    variable = elvol_aux_var
    execute_on = 'INITIAL'
  []

  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'Wall'
    variable = yw_aux_var
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
    type = FunctionIC
    variable = v
    function = v_start
  []

  [w_ic]
    type = ConstantIC
    variable = w
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
    type = INSFVInletVelocityBC
    boundary = 'Inlet'
    variable = u
    functor = 0
  []

  [inlet-v]
    type = INSFVInletVelocityBC
    boundary = 'Inlet'
    variable = v
    functor = 'v_in'
  []

  [inlet-w]
    type = INSFVInletVelocityBC
    boundary = 'Inlet'
    variable = w
    functor = 0
  []

  [wall-u]
    type = INSFVNoSlipWallBC
    boundary = 'Wall'
    variable = u
    function = 0
  []

  [wall-v]
    type = INSFVNoSlipWallBC
    boundary = 'Wall'
    variable = v
    function = 0
  []

  [wall-w]
    type = INSFVNoSlipWallBC
    boundary = 'Wall'
    variable = w
    function = 0
  []

  [outlet-p]
    type = INSFVOutletPressureBC
    boundary = 'Outlet'
    variable = pressure
    function = 0
  []
[]

[VectorPostprocessors]
  [vpp]
    type = ElementValueSampler
    variable = 'u v w pressure mixing_length_aux_var eddy_viscosity_aux_var yw_aux_var'
    sort_by = x
    execute_on = 'FINAL'
  []

  [elv]
    type = ElementValueSampler
    variable = 'elvol_aux_var yw_aux_var'
    sort_by = x
    execute_on = 'INITIAL'
  []
[]

[Preconditioning]
  [SMP_PJFNK]
    type = SMP
    full = true
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type -ksp_gmres_restart'
    petsc_options_value = 'lu 100'
  []
[]

[Executioner]
  type = Transient

  [TimeStepper]
    type = IterationAdaptiveDT
    dt = 1e-5
    growth_factor = 1.25
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.75
    optimal_iterations = 8
    linear_iteration_ratio = 150
  []

  start_time = 0
  end_time = 5000
  num_steps = 10000

  dtmin = 1e-8
  dtmax = 50

  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
  nl_max_its = 50

  l_tol = 1e-5
  l_max_its = 100

  steady_state_detection = true
  steady_state_tolerance = 1e-6
[]

[Outputs]
  print_linear_residuals = false

  [exodus]
    type = Exodus
    execute_on = FINAL
    file_base = migadome_3d_ml_out
  []

  [csv]
    type = CSV
    execute_on = FINAL
    file_base = migadome_3d_ml_csv
  []

  [checkpoint]
    type = Checkpoint
    execute_on = FINAL
    file_base = migadome_3d_ml_out
  []
[]