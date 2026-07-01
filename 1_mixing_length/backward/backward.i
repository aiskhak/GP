# 2D backward-facing step, standard mixing-length baseline
# Matching CTTC DNS: Re_tau = 395, ER = 2
#
# Non-dimensionalization:
#   h      = 1
#   u_tau  = 1
#   rho    = 1
#   Re_tau = u_tau h / nu = 395
#   mu     = 1/395
#
# DNS/domain convention:
#   x = -6..32
#   y = -1..1
#   step corner at x = 0, y = 0
#
# Expected mesh boundaries:
#   inlet, outlet, top_wall, bottom_wall, step_wall

# global parameters
mu = 2.531645569620253e-3   # 1/395, using u_tau = 1 and h = 1
rho = 1

mesh = '../../mesh/BFS_Ret395_ER2_uniform.msh'

advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

# Standard mixing-length parameters
kappa_ml = 0.41
delta_ml = 1.0    # start with step height h = 1 as outer scale

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
  # Planar 2D BFS: no RZ coordinate system.
  [./mesh_file]
    type = FileMeshGenerator
    file = ${mesh}
  []
[]

[Problem]
  fv_bcs_integrity_check = false
  # For first debugging runs, avoid restart.
  # restart_file_base = bfs_2d_fv_out_cp/LATEST
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

  [eddy_viscosity_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [dudx_aux_var]
    type = MooseVariableFVReal
  []

  [dudy_aux_var]
    type = MooseVariableFVReal
  []

  [dvdx_aux_var]
    type = MooseVariableFVReal
  []

  [dvdy_aux_var]
    type = MooseVariableFVReal
  []

  [strain_mag_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [omega_mag_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [omega_over_s_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [elvol_aux_var]
    order = CONSTANT
    family = MONOMIAL
  []

  [yw_aux_var]
    order = CONSTANT
    family = MONOMIAL
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
  [mixing_len_aux_ker]
    type = WallDistanceMixingLengthAux
    walls = 'top_wall bottom_wall step_wall'
    variable = mixing_length_aux_var
    von_karman_const = ${kappa_ml}
    delta = ${delta_ml}
  []
  [eddy_viscosity_aux_ker]
    type = INSFVMixingLengthTurbulentViscosityAux
    variable = eddy_viscosity_aux_var
    mixing_length = mixing_length_aux_var
    u = u
    v = v
  []
  [elvol_aux_ker]
    type = VolumeAux
    variable = elvol_aux_var
    execute_on = 'INITIAL'
  []
  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'top_wall bottom_wall step_wall'
    variable = yw_aux_var
    execute_on = 'INITIAL'
  []
  [dudx_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dudx_aux_var
    functor = grad_u
    component = 0
    execute_on = 'TIMESTEP_END FINAL'
  []

  [dudy_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dudy_aux_var
    functor = grad_u
    component = 1
    execute_on = 'TIMESTEP_END FINAL'
  []

  [dvdx_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dvdx_aux_var
    functor = grad_v
    component = 0
    execute_on = 'TIMESTEP_END FINAL'
  []

  [dvdy_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dvdy_aux_var
    functor = grad_v
    component = 1
    execute_on = 'TIMESTEP_END FINAL'
  []

  [strain_mag_aux_ker]
    type = StrainRotationFromGradAux
    variable = strain_mag_aux_var
    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var
    quantity = strain_mag
    execute_on = 'TIMESTEP_END FINAL'
  []

  [omega_mag_aux_ker]
    type = StrainRotationFromGradAux
    variable = omega_mag_aux_var
    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var
    quantity = omega_mag
    execute_on = 'TIMESTEP_END FINAL'
  []

  [omega_over_s_aux_ker]
    type = StrainRotationFromGradAux
    variable = omega_over_s_aux_var
    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var
    quantity = omega_over_s
    epsilon = 1e-12
    execute_on = 'TIMESTEP_END FINAL'
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
    functor  = 'u_in'
  []

  [inlet-v]
    type     = INSFVInletVelocityBC
    boundary = 'inlet'
    variable = v
    functor  = 0
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

[Materials]
  [grad_u_mat]
    type = ADGenericFunctorGradientMaterial
    prop_names = 'grad_u'
    prop_values = 'u'
  []

  [grad_v_mat]
    type = ADGenericFunctorGradientMaterial
    prop_names = 'grad_v'
    prop_values = 'v'
  []
[]

[VectorPostprocessors]
  [vpp]
    type = ElementValueSampler
    variable = 'u v pressure mixing_length_aux_var eddy_viscosity_aux_var yw_aux_var dudx_aux_var dudy_aux_var dvdx_aux_var dvdy_aux_var strain_mag_aux_var omega_mag_aux_var omega_over_s_aux_var'
    sort_by = x
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
    growth_factor = 1.25
    optimal_iterations = 8
    linear_iteration_ratio = 150
    dt = 1e-4
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.75
  []

  dtmin = 1e-8
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
  steady_state_tolerance = 1.e-6
[]

[Outputs]
  print_linear_residuals = false

  [exodus]
    type = Exodus
    execute_on = FINAL
    file_base = bfs_2d_fv_out
  []

  [csv]
    type = CSV
    execute_on = FINAL
    file_base = bfs_2d_fv_csv
  []

  [out]
    type = Checkpoint
    execute_on = FINAL
    file_base = bfs_2d_fv_out
  []
[]