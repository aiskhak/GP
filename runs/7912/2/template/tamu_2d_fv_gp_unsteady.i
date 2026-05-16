# global parameters
mu    = 1.263903e-4	# 1/7912
mesh  = '../../../../mesh/TAMU_2D_RANS_2.msh'
csv_f = 'lm_pred.csv'
csv_f_elem = 'elem_id.csv'

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
  [mixing_length_gp_aux_ker]
    type = AuxVarFromCSVFile
    variable = mixing_length_gp_aux_var
    file_name = ${csv_f}
    elem_id_file_name = ${csv_f_elem}
    use_mapping = true
    header = true
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

[VectorPostprocessors]
  [vpp]
    type = ElementValueSampler
    variable = 'u v'
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