# global parameters
mu    = 9.414423e-5	# 1/10622
mesh  = '../../../../mesh/TAMU_2D_RANS_4.msh'
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
    expression = -1*(60/49)*(1-y/0.5)^(1/7)
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
    mu = mu_eff
    momentum_component = 'x'
    complete_expansion = true
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
    mu = mu_eff
    momentum_component = 'y'
    complete_expansion = true
    u = u
    v = v
  []

  [v_viscosity_rz]
    type = INSFVMomentumViscousSourceRZ
    variable = v
    mu = mu_eff
    momentum_component = 'y'
    complete_expansion = true
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



[FunctorMaterials]
  [mixing_length_viscosity]
    type = INSFVMixingLengthEffectiveViscosityFunctorMaterialRZ
    property_name = mu_eff
    turbulent_viscosity_property_name = mu_t
    molecular_viscosity = ${mu}
    rho = ${rho}
    mixing_length = mixing_length_gp_aux_var
    u = u
    v = v
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
    mu       = mu_eff
    momentum_component = x
  []
  [axis-v]
    type     = INSFVSymmetryVelocityBC
    boundary = 'SYM'
    variable = v
    u        = u
    v        = v
    mu       = mu_eff
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

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  nl_abs_tol          = 1e-6
  nl_rel_tol          = 1e-6
  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'preonly lu       superlu_dist'
  automatic_scaling   = true
  nl_max_its          = 20
[]

[Outputs]
  print_linear_residuals = false

  #[exodus]
  #  type = Exodus
  #  execute_on = FINAL
  #  file_base = tamu_2d_fv_gp_out
  #[]

  [csv]
    type = CSV
    execute_on = FINAL
    file_base = tamu_2d_fv_gp_csv
  []

  [out]
    type = Checkpoint
    execute_on = FINAL
    file_base = tamu_2d_fv_gp_out
  []
[]