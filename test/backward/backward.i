# global parameters
mu = 2.531645569620253e-3   # 1/395, using u_tau = 1 and h = 1
rho = 1

mesh = '../../mesh/BFS_Ret395_ER2_uniform.msh'

advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

# GP-corrected mixing-length parameters
# Same discovered law as Migadome test:
#   raw = tanh(-0.08485482588769211)
#   eta_y = yw / D0_eta
#   lm_gp = lm_std * factor
#
# For BFS Retau=395 setup:
#   h = 1, so use delta_ml = 1.0 for the baseline mixing-length cap
#   D0_eta = 1.0 for eta_y normalization

kappa_ml = 0.41
delta_ml = 1.0
D0_eta = 1.0

activation_eta_y_gp = 0.5
correction_amplitude_gp = 0.8
gp_const = 0.08485482588769211

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
  [./mesh_file]
    type = FileMeshGenerator
    file = ${mesh}
  []
[]

[Problem]
  fv_bcs_integrity_check = false
  #restart_file_base = bfs_2d_fv_gp_out_cp/LATEST
  allow_initial_conditions_with_restart = true
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
  # GP-corrected mixing length used by the Reynolds-stress kernels.
  [mixing_length_gp_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  # Diagnostic: baseline wall-distance mixing length.
  [mixing_length_std_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  # Diagnostic: GP multiplier applied to the baseline mixing length.
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

  [yw_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []
[]

[Functions]
  [u_in]
    type = PiecewiseLinear
    data_file = 'dns_inlet_u_xminus5p98.csv'
    format = columns
    axis = y
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
    variable = 'u'
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
  mixing_length = mixing_length_gp_aux_var
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
  mixing_length = mixing_length_gp_aux_var
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
  # Wall distance used for eta_y and diagnostics
  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'top_wall bottom_wall step_wall'
    variable = yw_aux_var
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  # Baseline standard mixing length:
  # lm_std = WallDistanceMixingLengthAux(kappa_ml, delta_ml)
  # For BFS, delta_ml = 1.0 because h = 1.
  [mixing_len_std_aux_ker]
    type = WallDistanceMixingLengthAux
    walls = 'top_wall bottom_wall step_wall'
    variable = mixing_length_std_aux_var
    von_karman_const = ${kappa_ml}
    delta = ${delta_ml}
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  # GP multiplier:
  # raw = tanh(-gp_const)
  # eta_y = yw / D0_eta
  # factor = 1 for eta_y < 0.5
  # factor = 1 + 0.8*tanh((eta_y - 0.5)*raw) otherwise
  [gp_factor_aux_ker]
    type = ParsedAux
    variable = gp_factor_aux_var
    coupled_variables = 'yw_aux_var'
    expression = 'if(yw_aux_var/${D0_eta} < ${activation_eta_y_gp}, 1.0, 1.0 + ${correction_amplitude_gp}*tanh((yw_aux_var/${D0_eta} - ${activation_eta_y_gp})*tanh(-${gp_const})))'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  # Final GP-corrected mixing length:
  # lm_gp = lm_std * factor
  [mixing_length_gp_aux_ker]
    type = ParsedAux
    variable = mixing_length_gp_aux_var
    coupled_variables = 'mixing_length_std_aux_var gp_factor_aux_var'
    expression = 'mixing_length_std_aux_var * gp_factor_aux_var'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []

  [eddy_viscosity_aux_ker]
    type = INSFVMixingLengthTurbulentViscosityAux
    variable = eddy_viscosity_aux_var
    mixing_length = mixing_length_gp_aux_var
    u = u
    v = v
    execute_on = 'TIMESTEP_END FINAL'
  []
[]

[Postprocessors]
  [eta_y_scale_pp]
    type = ElementExtremeValue
    variable = yw_aux_var
    value_type = max
    execute_on = 'INITIAL'
  []
[]

[ICs]
  [u_ic]
    type = FunctionIC
    variable = u
    function = u_in
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
    functor = 'u_in'
  []
  [inlet-v]
    type     = INSFVInletVelocityBC
    boundary = 'inlet'
    variable = v
    functor = 0
  []
	[top-wall-u]
	  type     = INSFVNoSlipWallBC
	  boundary = 'top_wall'
	  variable = u
	  function = 0
	[]

	[top-wall-v]
	  type     = INSFVNoSlipWallBC
	  boundary = 'top_wall'
	  variable = v
	  function = 0
	[]

	[bottom-wall-u]
	  type     = INSFVNoSlipWallBC
	  boundary = 'bottom_wall'
	  variable = u
	  function = 0
	[]

	[bottom-wall-v]
	  type     = INSFVNoSlipWallBC
	  boundary = 'bottom_wall'
	  variable = v
	  function = 0
	[]

	[step-wall-u]
	  type     = INSFVNoSlipWallBC
	  boundary = 'step_wall'
	  variable = u
	  function = 0
	[]

	[step-wall-v]
	  type     = INSFVNoSlipWallBC
	  boundary = 'step_wall'
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
    variable = 'u v pressure mixing_length_gp_aux_var mixing_length_std_aux_var gp_factor_aux_var eddy_viscosity_aux_var yw_aux_var'
    sort_by = x
  []
[]

[Preconditioning]
  [./SMP_PJFNK]
    type = SMP
    full = true
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type -ksp_gmres_restart'
    petsc_options_value = 'lu 100'
  [../]
[]

[Executioner]
  type = Transient
  [./TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.25
    optimal_iterations = 8
    linear_iteration_ratio = 150
    dt = 1e-4
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.75
  [../]
  dtmin = 1e-8
  dtmax = 50
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
  nl_max_its = 50
  l_tol = 1e-5
  l_max_its = 100
  start_time = 0
  end_time  = 10000
  num_steps = 10000
  steady_state_detection = true
  steady_state_tolerance = 1.e-6
[]

[Outputs]
  print_linear_residuals = false
  [exodus]
    type = Exodus
    execute_on = FINAL
	file_base = bfs_2d_fv_gp_out
  []
  [./csv]
    type = CSV
	execute_on = FINAL
	file_base = bfs_2d_fv_gp_csv
  []
  [./out]
    type = Checkpoint
    execute_on = FINAL
    file_base = bfs_2d_fv_gp_out
  []
[]