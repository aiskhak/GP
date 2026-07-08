# Cervi test1 tank, standard mixing-length baseline, dx=0.025
#
# Purpose:
#   Coarse-grid MOOSE FV mixing-length case for comparison against the
#   OpenFOAM SST quasi-steady velocity reference:
#
#   /homes/aiskhak/projects/GP/0_data_les/cervi/test1_sst_simpleFoam_qs/20000/U
#
# OpenFOAM reference setup:
#   solver: simpleFoam
#   model: incompressible k-omega SST
#   rho = 927 kg/m^3
#   mu  = 1.84e-5 Pa s
#   nu  = mu/rho = 1.984897519e-8 m^2/s
#   inlet velocity = (0, 0.05) m/s
#   outlet pressure = 0
#   walls no-slip
#
# MOOSE geometry:
#   x = 0..1 tank, outlet extends to x = 1.4
#   y = 0..1 tank, inlet extends to y = -0.4
#   inlet:  x = 0.30..0.50 at y = -0.40
#   outlet: y = 0.50..0.70 at x = 1.40
#
# Mesh boundaries:
#   inlet, outlet, walls

rho = 927
mu = 1.84e-5

mesh = 'mesh/cervi_test1_dx025.msh'

advected_interp_method = 'upwind'
velocity_interp_method = 'rc'

# Standard mixing-length parameters
# l_m = min(kappa*y_w, 0.09*delta)
# Cervi inlet width = 0.20 m, so delta = half inlet width = 0.10 m
kappa_ml = 0.41
delta_ml = 0.10

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
  [mixing_len_aux_ker]
    type = WallDistanceMixingLengthAux
    walls = 'walls'
    variable = mixing_length_aux_var
    von_karman_const = ${kappa_ml}
    delta = ${delta_ml}
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

  [yw_aux_ker]
    type = WallDistanceAux
    walls = 'walls'
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
    variable = 'u v pressure mixing_length_aux_var eddy_viscosity_aux_var'
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
    file_base = cervi_ml_dx025_delta0p1
  []

  [csv]
    type = CSV
    execute_on = FINAL
    file_base = cervi_ml_dx025_delta0p1
  []

  [out]
    type = Checkpoint
    execute_on = FINAL
    file_base = cervi_ml_dx025_delta0p1_cp
  []
[]
