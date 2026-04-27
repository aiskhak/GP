# global parameters
mu    = 7.142857e-5	# 1/14000
mesh  = '../../../mesh/TAMU_2D_RANS_2.msh'

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
  restart_file_base = tamu_2d_fv_out_cp/LATEST
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
  [elvol_aux_var]
    order = CONSTANT
    family = MONOMIAL
  []
  [yw_aux_var]	# distance to the nearest wall
    order = CONSTANT
    family = MONOMIAL
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
    walls = 'wall'
    variable = mixing_length_aux_var
    von_karman_const = 0.41
    delta = 0.5
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
	execute_on = 'initial'
  []
  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'wall'
    variable = yw_aux_var
    execute_on = 'initial'
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
    variable = 'u v eddy_viscosity_aux_var'
    sort_by = x
  []
  [elv]
    type = ElementValueSampler
    variable = 'elvol_aux_var yw_aux_var'
    sort_by = x
    execute_on = 'initial'
  []
[]

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  nl_abs_tol          = 1e-6
  nl_rel_tol          = 1e-6
  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'preonly lu       superlu_dist'
  automatic_scaling   = true
  nl_max_its          = 100
[]

[Outputs]
  print_linear_residuals = false
  [exodus]
    type = Exodus
    execute_on = FINAL
	file_base = tamu_2d_fv_out
  []
  [./csv]
    type = CSV
	execute_on = FINAL
	file_base = tamu_2d_fv_csv
  []
  [./out]
    type = Checkpoint
    execute_on = FINAL
    file_base = tamu_2d_fv_out
  []
[]