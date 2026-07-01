# global parameters
mu    = 3.333333e-3	# 1/3000
mesh  = '../../../../mesh/TAMU_2D_RANS_3.msh'

rho = 1
advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

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
  coord_type = 'RZ'
  rz_coord_axis = x
  [./mesh_file]
    type = FileMeshGenerator
    file = ${mesh}
  []
  [./scale]
    type = TransformGenerator
    input = mesh_file
    transform = SCALE
    vector_value ='0.05249344 0.05249344 0.05249344'  # 1/19.05
  []
[]

[Problem]
  fv_bcs_integrity_check = false
  restart_file_base = tamu_2d_fv_gp_out_cp/LATEST
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
  [mixing_length_gp_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [yw_aux_var]
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

  [strain_factor_raw_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [strain_factor_clipped_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [strain_factor_clip_delta_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []

  [strain_invariant_aux_var]
    order  = CONSTANT
    family = MONOMIAL
    fv     = true
  []
[]

[Functions]
  [./u_in]
    type = ParsedFunction
    expression = -1*(8/7)*(1-y/0.5)^(1/7)
  [../]
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
    type = INSFVMixingLengthReynoldsStress_gp
    variable = u
    rho = ${rho}
    mixing_length_gp = mixing_length_gp_aux_var
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
    type = INSFVMixingLengthReynoldsStress_gp
    variable = v
    rho = ${rho}
    mixing_length_gp = mixing_length_gp_aux_var
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
    walls = 'wall'
    variable = yw_aux_var
    execute_on = 'INITIAL'
  []

  [dudx_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dudx_aux_var
    functor = grad_u
    component = 0
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [dudy_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dudy_aux_var
    functor = grad_u
    component = 1
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [dvdx_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dvdx_aux_var
    functor = grad_v
    component = 0
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [dvdy_aux_ker]
    type = ADFunctorVectorElementalAux
    variable = dvdy_aux_var
    functor = grad_v
    component = 1
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [mixing_length_gp_aux_ker]
    type = DynamicStrainGPBlendedMixingLengthAux
    variable = mixing_length_gp_aux_var
    output_quantity = mixing_length

    expression_file = closure_expr.txt

    wall_distance = yw_aux_var

    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var

    kappa = 0.41
    kappa_cap = 0.09
    delta0 = 1.0

    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [strain_factor_raw_aux_ker]
    type = DynamicStrainGPBlendedMixingLengthAux
    variable = strain_factor_raw_aux_var
    output_quantity = strain_factor_raw

    expression_file = closure_expr.txt

    wall_distance = yw_aux_var

    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var

    kappa = 0.41
    kappa_cap = 0.09
    delta0 = 1.0

    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [strain_factor_clipped_aux_ker]
    type = DynamicStrainGPBlendedMixingLengthAux
    variable = strain_factor_clipped_aux_var
    output_quantity = strain_factor_clipped

    expression_file = closure_expr.txt

    wall_distance = yw_aux_var

    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var

    kappa = 0.41
    kappa_cap = 0.09
    delta0 = 1.0

    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [strain_factor_clip_delta_aux_ker]
    type = DynamicStrainGPBlendedMixingLengthAux
    variable = strain_factor_clip_delta_aux_var
    output_quantity = strain_factor_clip_delta

    expression_file = closure_expr.txt

    wall_distance = yw_aux_var

    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var

    kappa = 0.41
    kappa_cap = 0.09
    delta0 = 1.0

    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
  []

  [strain_invariant_aux_ker]
    type = DynamicStrainGPBlendedMixingLengthAux
    variable = strain_invariant_aux_var
    output_quantity = strain_invariant

    expression_file = closure_expr.txt

    wall_distance = yw_aux_var

    dudx = dudx_aux_var
    dudy = dudy_aux_var
    dvdx = dvdx_aux_var
    dvdy = dvdy_aux_var

    kappa = 0.41
    kappa_cap = 0.09
    delta0 = 1.0

    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END FINAL'
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
  [no-slip-wall-u]
    type     = INSFVNoSlipWallBC
    boundary = 'wall'
    variable = u
    function = 0
  []
  [no-slip-wall-v]
    type     = INSFVNoSlipWallBC
    boundary = 'wall'
    variable = v
    function = 0
  []
  [outlet-p]
    type     = INSFVOutletPressureBC
    boundary = 'outlet'
    variable = pressure
    function = 0
  []
  [axis-u]
    type     = INSFVSymmetryVelocityBC
    boundary = 'SYM'
    variable = u
    u        = u
    v        = v
    mu       = ${mu}
    momentum_component = x
  []
  [axis-v]
    type     = INSFVSymmetryVelocityBC
    boundary = 'SYM'
    variable = v
    u        = u
    v        = v
    mu       = ${mu}
    momentum_component = y
  []
  [axis-p]
    type     = INSFVSymmetryPressureBC
    boundary = 'SYM'
    variable = pressure
  []
[]

[FunctorMaterials]
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
    variable = 'u v strain_factor_raw_aux_var strain_factor_clipped_aux_var'
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
    dt = 0.5
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.75
  [../]
  dtmin = 1e-6
  dtmax = 200
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
#  [exodus]
#    type = Exodus
#    execute_on = FINAL
#	file_base = tamu_2d_fv_gp_out
#  []
  [./csv]
    type = CSV
	execute_on = FINAL
	file_base = tamu_2d_fv_gp_csv
  []
  [./out]
    type = Checkpoint
    execute_on = FINAL
    file_base = tamu_2d_fv_gp_out
  []
[]