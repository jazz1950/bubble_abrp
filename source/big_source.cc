#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <Epetra_Map.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <string>

  using namespace dealii;

  class ParameterReader : public Subscriptor
  {
    public:
      ParameterReader(ParameterHandler &);
      void read_parameters(const std::string);

    private:
      void declare_parameters();
      ParameterHandler &prm;
  };

  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    :
  prm(paramhandler)
  {}

  void ParameterReader::declare_parameters()
  {
    prm.enter_subsection ("Mesh & geometry parameters");
      prm.declare_entry ("Gmesh input" , "false", Patterns::Bool());
      prm.declare_entry ("Input file name", "",Patterns::Anything());
      prm.declare_entry ("X-axis min" , "-0.5" , Patterns::Double(-100, 100));
      prm.declare_entry ("X-axis max" , "+0.5" , Patterns::Double(-100, 100));
      prm.declare_entry ("Y-axis min" , "-0.5" , Patterns::Double(-100, 100));
      prm.declare_entry ("Y-axis max" , "+0.5" , Patterns::Double(-100, 100));
      prm.declare_entry ("Z-axis min" , "-0.5" , Patterns::Double(-100, 100));
      prm.declare_entry ("Z-axis max" , "+0.5" , Patterns::Double(-100, 100));
    prm.leave_subsection ();

    prm.enter_subsection ("Entropy Viscosity");
      prm.declare_entry ("Beta", "1", Patterns::Double(0.0,100.));
      prm.declare_entry ("c_R factor", "1", Patterns::Double(0.0,100.));
    prm.leave_subsection ();

    prm.enter_subsection ("Boundary conditions");
      prm.declare_entry ("Boundary for X-axis min", "0", Patterns::Integer(0,100));
      prm.declare_entry ("Boundary for X-axis max", "0", Patterns::Integer(0,100));
      prm.declare_entry ("Boundary for Y-axis min", "0", Patterns::Integer(0,100));
      prm.declare_entry ("Boundary for Y-axis max", "0", Patterns::Integer(0,100));
      prm.declare_entry ("Boundary for Z-axis min", "0", Patterns::Integer(0,100));
      prm.declare_entry ("Boundary for Z-axis max", "0", Patterns::Integer(0,100));
    prm.leave_subsection ();

    prm.enter_subsection ("Adapative Mesh Refinement");
      prm.declare_entry ("Global level", "0", Patterns::Integer(0,9));
      prm.declare_entry ("Local level" , "0" , Patterns::Integer(0,9));
      prm.declare_entry ("Droplet adaptive factor" , "0" , Patterns::Double(0,100));
      prm.declare_entry ("Safe region" , "0" , Patterns::Double(0,100));
    prm.leave_subsection ();

    prm.enter_subsection ("Equation Data");
      prm.declare_entry ("Viscosity of liquid" , "0.5" , Patterns::Double(0.0, 100000));
      prm.declare_entry ("Viscosity of droplet" , "0.5" , Patterns::Double(0.0, 100000));
      prm.declare_entry ("Density of liquid" , "0.5" , Patterns::Double(0.0, 100000));
      prm.declare_entry ("Density of droplet" , "0.5" , Patterns::Double(0.0, 100000));
      prm.declare_entry ("Density of particle" , "0.5" , Patterns::Double(0.0, 100000));
      prm.declare_entry ("Surface tension" , "1" , Patterns::Double(0.0, 100000));
    prm.leave_subsection ();


    prm.enter_subsection ("Droplet Information");
      prm.declare_entry ("Droplet radius", "0.1", Patterns::Double(0, 100));
      prm.declare_entry ("X-coord", "+0.0" , Patterns::Double(-100, 100));
      prm.declare_entry ("Y-coord", "+0.0" , Patterns::Double(-100, 100));
      prm.declare_entry ("Z-coord", "+0.0" , Patterns::Double(-100, 100));
      prm.declare_entry ("Droplet real diameter", "0.001", Patterns::Double(0, 100));
      prm.declare_entry ("Type for thickness" , "0" , Patterns::Integer(0,10));
      prm.declare_entry ("Factor for thickness", "1", Patterns::Double(0, 100));
    prm.leave_subsection ();

    prm.enter_subsection ("Mass Transfer");
      prm.declare_entry ("Constant mode inside", "false" , Patterns::Bool());
      prm.declare_entry ("Ratio Solubility", "0.1", Patterns::Double(0, 100));
      prm.declare_entry ("Diffusivity liquid", "+0.0" , Patterns::Double(-100, 100));
      prm.declare_entry ("Diffusivity droplet", "+0.0" , Patterns::Double(-100, 100));
    prm.leave_subsection ();
    
    prm.enter_subsection ("Particle Information");
      prm.declare_entry ("Type for initial distribution" , "0" , Patterns::Integer(0,10));
      prm.declare_entry ("Number of particle" , "0" , Patterns::Integer(0,10000));
      prm.declare_entry ("Diameter" , "0.0" , Patterns::Double(0.0,100));
      prm.declare_entry ("Factor for viscosity" , "0.0" , Patterns::Double(0.0,100000));
      prm.declare_entry ("Sedimentation" , "false" , Patterns::Bool());
      prm.declare_entry ("Distance from droplet", "0.0", Patterns::Double(-100,100));
    prm.leave_subsection ();
    
    prm.enter_subsection ("Period");
      prm.declare_entry ("output period" , "0" , Patterns::Integer(0,10000));
      prm.declare_entry ("refine period" , "-1" , Patterns::Integer(-1,10000));
    prm.leave_subsection ();

    prm.enter_subsection ("Problem Definition");
      prm.declare_entry ("Verbal output" , "false" , Patterns::Bool());
      prm.declare_entry ("Dimension" , "2" , Patterns::Integer(1,3));
      prm.declare_entry ("Error NS" , "1e-06" , Patterns::Double(0,1));
      prm.declare_entry ("Error level set" , "1e-06" , Patterns::Double(0,1));
      prm.declare_entry ("Error mass transfer" , "1e-11" , Patterns::Double(0,1));
      prm.declare_entry ("Error reinit" , "1e-3" , Patterns::Double(0,1));
      prm.declare_entry ("CFL number" , "0.1" , Patterns::Double(0.000,1.000));
      prm.declare_entry ("Time interval", "0.0", Patterns::Double(0, 1));
      prm.declare_entry ("End time", "1.0", Patterns::Double(0, 100));
      prm.declare_entry ("Buffered interval" , "1" , Patterns::Integer(1,10000));
    prm.leave_subsection ();
  }

  void ParameterReader::read_parameters(const std::string parameter_file)
  {
    declare_parameters();
    prm.read_input (parameter_file);
  }

  template <int dim>
  class ini_levelset_function : public Function<dim>
  {
    public:

    ini_levelset_function (Point<dim> drp_pos, double drp_radius, double epsilon);
    virtual double value (const Point<dim>   &p,
     const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
    Vector<double>   &value) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &p,
        std::vector<Vector<double> > &values) const;
    Point<dim> drp_pos;
    double drp_radius, epsilon;
  };

  template <int dim>
  ini_levelset_function<dim>::ini_levelset_function (Point<dim> drp_pos, 
            double drp_radius, 
            double epsilon):
  Function<dim>(1),drp_pos(drp_pos), drp_radius(drp_radius), epsilon(epsilon)
  {}

  template <int dim>
  double ini_levelset_function<dim>::value (const Point<dim>  &p,
         const unsigned int component) const
  {
    double dist_cor = 0.0;
    dist_cor =  drp_pos.distance(p) - drp_radius;
    return 1./(1. + std::exp(dist_cor/epsilon));
  }

  template <int dim>
  void ini_levelset_function<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = ini_levelset_function<dim>::value (p, c);
  }

  template <int dim>
  void ini_levelset_function<dim>::vector_value_list (const std::vector<Point<dim> > &points,
       std::vector<Vector<double> >   &value_list) const
  {
    for (unsigned int p=0; p<points.size(); ++p)
      ini_levelset_function<dim>::vector_value (points[p], value_list[p]);
  }

  template <int dim>
  class initial_concentration : public Function<dim>
  {
    public:

    initial_concentration ( Point<dim> coord, 
       double radius,
       double aaa, 
       double ratio_soluv);

    virtual double value (const Point<dim>   &p,
     const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
    Vector<double>   &value) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &p,
        std::vector<Vector<double> > &values) const;

    Point<dim> coord;
    double radius;
    double aaa;
    double ratio_soluv;
  };

  template <int dim>
  initial_concentration<dim>::initial_concentration (Point<dim> coord, 
            double radius,
            double aaa, 
            double ratio_soluv) :
    Function<dim>(1),
    coord(coord),
    radius(radius),
    aaa (aaa),
    ratio_soluv (ratio_soluv)
    {}

  template <int dim>
  double initial_concentration<dim>::value (const Point<dim>  &p,
         const unsigned int component) const
  {
    double zz = 0.0;
    double bb = coord.distance(p) - radius;
    if (bb > 0.0) zz = 0.0*std::sqrt(ratio_soluv);
    if (bb < 0.0) zz = aaa/std::sqrt(ratio_soluv);
    return zz;
  }

  template <int dim>
  void initial_concentration<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = initial_concentration<dim>::value (p, c);
  }

  template <int dim>
  void initial_concentration<dim>::vector_value_list (const std::vector<Point<dim> > &points,
            std::vector<Vector<double> >   &value_list) const
  {
    for (unsigned int p=0; p<points.size(); ++p)
      initial_concentration<dim>::vector_value (points[p], value_list[p]);
  }
  
  template <int dim>
  class AdvectionDiffusion
  {
    public:
      AdvectionDiffusion (ParameterHandler &);
      ~AdvectionDiffusion ();
    void run ();

    private:

    void readat ();
    void create_init_coarse_mesh ();
    void create_init_adaptive_mesh ();
    void refine_mesh ();
    void prescribe_flag_refinement ();
    void adp_execute_transfer ();
    void particle_amr_loop ();
    void levelset_amr_loop ();     
    
    void setup_dofs (unsigned int);

    void diffusion_step ();
    void projection_step ();
    void pressure_correction_step_rot ();
    void solution_update ();
    void vel_pre_convergence ();
    double get_maximal_velocity () const;
    double determine_time_step ();
    
    void initial_Levelset ();
    void prepare_for_levelset ();
    void level_set_2nd_adv_step ();
    void solve_for_adv_step ();
    void level_set_compute_normal (unsigned int);
    void sufTen_compute_gradient (unsigned int);
    void surTen_compute_curvature ();
    void level_set_reinitial_step ();

    void initial_point_concentration ();
    void assemble_concentr_system ();
    void concentr_solve ();
    void recover_org_concentration ();
    void set_constant_mode_concentration ();
    
    void initial_particle_distribution ();
    void particle_solution ();
    std::pair<unsigned int, double> DistPar (Point<dim> &, 
       unsigned int number_of_particles);
    void particle_generation ();
    void pars_move (std::ofstream &);
    
    void initialize_for_mass_transfer_parameter ();
    void droplet_parameter (std::ofstream &);
    void mass_transfer_parameter (std::vector<Point<dim> > &, 
      std::ofstream &);
    void compute_normal_grad_mass (std::vector<bool> &);
    void find_midPoint_on_interCell (std::vector<bool> &,
          std::vector<Point<dim> > &);
    
    void solve_fluid_equation ();
    void solve_levelSet_function ();
    void solve_mass_transfer_equation ();
    
    void plotting_solution (unsigned int);
    std::pair<double,double> get_extrapolated_levelSet_range ();
    std::pair<double,double> get_extrapolated_concentr_range ();
    double get_entropy_variation_levelset (const double average_levelset) const;
    double get_entropy_variation_concentr (const double average_concentr) const;

    double
    compute_viscosity(const std::vector<double>           &levelSet,
      const std::vector<double>           &old_levelSet,
      const std::vector<Tensor<1,dim> >   &levelSet_grads,
      const std::vector<Tensor<1,dim> >   &old_levelSet_grads,
      const std::vector<double>           &levelSet_laplacians,
      const std::vector<double>           &old_levelSet_laplacians,
      const std::vector<Tensor<1,dim> >   &velocity_values,
      const std::vector<Tensor<1,dim> >   &old_velocity_values,
      const double                        global_u_infty,
      const double                        global_i_variation,
      const double       average_levelset,
      const double        global_entropy_variation,
      const double                         cell_diameter);

    std::pair<double,double>
    compute_entropy_viscosity_for_navier_stokes(
                              const std::vector<Tensor<1,dim> >     &old_velocity,
                              const std::vector<Tensor<1,dim> >     &old_old_velocity,
                              const std::vector<Tensor<2,dim> >     &old_velocity_grads,
                              const std::vector<Tensor<1,dim> >     &old_velocity_laplacians,
                              const std::vector<Tensor<1,dim> >     &old_pressure_grads,
                              const Tensor<1,dim>                   &source_vector,
                              const double                           coeff1_for_adv,
                              const double                           coeff2_for_visco,
                              const double                           coeff3_for_source,
                              const double                           cell_diameter) const;

      void compute_effective_viscosity (unsigned int iii, std::ofstream &out_q1);
      
      const Epetra_Comm    &trilinos_communicator;
      ConditionalOStream   pcout;
      ParameterHandler    &prm;
      Triangulation<dim>                   triangulation;

      const FESystem<dim>   fe_velocity;
      FE_Q<dim>     fe_pressure;
      FE_Q<dim>     fe_levelset;
      FE_Q<dim>     fe_concentr;

      DoFHandler<dim>                      dof_handler_velocity;
      DoFHandler<dim>                      dof_handler_pressure;
      DoFHandler<dim>                      dof_handler_levelset;
      DoFHandler<dim>                      dof_handler_concentr;

      ConstraintMatrix                     constraint_velocity;
      ConstraintMatrix                     constraint_pressure;
      ConstraintMatrix                     constraint_levelset;
      ConstraintMatrix                     constraint_concentr;

      TrilinosWrappers::SparseMatrix       matrix_velocity;
      TrilinosWrappers::SparseMatrix       matrix_pressure;
      TrilinosWrappers::SparseMatrix       matrix_levelset;
      TrilinosWrappers::SparseMatrix       matrix_concentr;

      TrilinosWrappers::Vector   vel_star, vel_n_plus_1, vel_n, vel_n_minus_1;
      TrilinosWrappers::Vector   aux_n_plus_1, aux_n, aux_n_minus_1;
      TrilinosWrappers::Vector   pre_star, pre_n_plus_1, pre_n, pre_n_minus_1;
      TrilinosWrappers::Vector   trn_concentr_on_pressure;
      TrilinosWrappers::Vector   vel_res, pre_res;

      TrilinosWrappers::Vector   level_set_curvature;
      TrilinosWrappers::Vector   levelset_solution, old_levelset_solution;
      TrilinosWrappers::Vector   level_set_grad_x, level_set_grad_y, level_set_grad_z;
      TrilinosWrappers::Vector   level_set_normal_x, level_set_normal_y, level_set_normal_z;
      TrilinosWrappers::Vector   particle_at_levelset;
      TrilinosWrappers::Vector   dimless_density_distribution, dimless_viscosity_distribution;

      TrilinosWrappers::Vector   concentr_solution, old_concentr_solution;
      TrilinosWrappers::Vector   org_concentration;
      TrilinosWrappers::Vector   normal_vec_out_flux_concentr, normal_vec_out_flux_concentr2;
      TrilinosWrappers::Vector   interface_on_cell;
      
      TrilinosWrappers::MPI::Vector  rhs_velocity;
      TrilinosWrappers::MPI::Vector  rhs_pressure;
      TrilinosWrappers::MPI::Vector  rhs_levelset;
      TrilinosWrappers::MPI::Vector  rhs_concentr;
      
      //Mesh & Geometry
      bool  iGmsh;
      std::string input_mesh_file;
      double  xmin, xmax, ymin, ymax, zmin, zmax;

      //Entropy Viscosity
      double  stabilization_beta, stabilization_c_R;

      //Boundary Condition;
      double  bnd_x_min, bnd_y_min, bnd_z_min;
      double  bnd_x_max, bnd_y_max, bnd_z_max;

      //AMR
      double  drop_adp_fac, safe_fac;
      unsigned int global_refinement_level, local_refinement_level;
      unsigned int  max_level, min_level;

      //Equation Data
      double  viscosity_liquid, viscosity_droplet;
      double  density_liquid, density_droplet;
      double  density_particle;
      double  surface_tension;

      //Droplet Information
      double  droplet_radius, droplet_real_diameter;
      double  droplet_x_coor, droplet_y_coor, droplet_z_coor;
      double  fac_thck;
      double  avr_droplet_size, deformability;
      Point<dim> avr_droplet_vel, avr_droplet_pos;
      unsigned int type_for_thck;

      //Period
      unsigned int output_fac, refine_fac;

      //Problem Definition
      bool  is_verbal_output;
      double  error_vel, error_levelset, error_concentr, error_reinit;
      double  cfl_number, end_time;

      //Level Set Function
      double  eps_v_levelset, tau_step, smallest_h_size;
      
      //Dimensionless Number
      double  time_ref, length_ref, velocity_ref, force_ref;
      double  Archimedes_number, Bond_number, Morton_num;
      
      //Particle Information
      unsigned int  type_init_distb;
      unsigned int  num_of_par;
      double  par_diameter;
      double  factor_visocisty;
      bool  is_sedimentaion;
      double  dist_from_droplet;
      std::vector<Point<dim> > particle_position;
      
      //Mass Transfer
      bool  is_set_const_mode;
      double  diffusivity_droplet, diffusivity_liquid;
      double  ratio_soluv;
      double  sum_of_mass_normal_flux1;
      double  sum_mass_in_liquid_at_interface;
      unsigned int  intercell_no, intercell_no_velNorm;
      
      //-----------------------------------------------------------
      double   time_step_upl, old_time_step_upl;
      unsigned int timestep_number; 
      double  total_time;
      double  maximal_velocity;
      unsigned int  no_buffered_interval;

  };

  template <int dim>
  AdvectionDiffusion<dim>::AdvectionDiffusion (ParameterHandler &param)
    :
  trilinos_communicator (Utilities::Trilinos::comm_world()),
  pcout (std::cout, Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0),
  prm(param),
  fe_velocity (FE_Q<dim>(2), dim),
  fe_pressure (1),
  fe_levelset (2),
  fe_concentr (1),
  dof_handler_velocity (triangulation),
  dof_handler_pressure (triangulation),
  dof_handler_levelset (triangulation),
  dof_handler_concentr (triangulation),
  time_step_upl (0),
  old_time_step_upl (0),
  timestep_number (0),
  total_time (0)
  {} 

  template <int dim>
  AdvectionDiffusion<dim>::~AdvectionDiffusion ()
  {
    dof_handler_velocity.clear ();
    dof_handler_pressure.clear ();
    dof_handler_levelset.clear ();
    dof_handler_concentr.clear ();
  }

  template <int dim>
  void AdvectionDiffusion<dim>::readat ()
  {
    pcout << "* Read Data.." << std::endl;
    
    prm.enter_subsection ("Mesh & geometry parameters");
      iGmsh = prm.get_bool ("Gmesh input");
      input_mesh_file = prm.get ("Input file name");
      xmin = prm.get_double ("X-axis min");
      xmax = prm.get_double ("X-axis max");
      ymin = prm.get_double ("Y-axis min");
      ymax = prm.get_double ("Y-axis max");
      zmin = prm.get_double ("Z-axis min");
      zmax = prm.get_double ("Z-axis max");
    prm.leave_subsection ();

    prm.enter_subsection ("Entropy Viscosity");
      stabilization_beta = prm.get_double ("Beta");
      stabilization_c_R = prm.get_double ("c_R factor");
    prm.leave_subsection ();

    prm.enter_subsection ("Boundary conditions");
      bnd_x_min = prm.get_integer ("Boundary for X-axis min");
      bnd_x_max = prm.get_integer ("Boundary for X-axis max");
      bnd_y_min = prm.get_integer ("Boundary for Y-axis min");
      bnd_y_max = prm.get_integer ("Boundary for Y-axis max");
      bnd_z_min = prm.get_integer ("Boundary for Z-axis min");
      bnd_z_max = prm.get_integer ("Boundary for Z-axis max");
    prm.leave_subsection ();

    prm.enter_subsection ("Adapative Mesh Refinement");
      global_refinement_level = prm.get_integer ("Global level");
      local_refinement_level = prm.get_integer ("Local level");
      drop_adp_fac = prm.get_double ("Droplet adaptive factor");
      safe_fac = prm.get_double ("Safe region");
    prm.leave_subsection ();

    prm.enter_subsection ("Equation Data");
      viscosity_liquid = prm.get_double ("Viscosity of liquid");
      viscosity_droplet = prm.get_double ("Viscosity of droplet");
      density_liquid = prm.get_double ("Density of liquid");
      density_droplet = prm.get_double ("Density of droplet");
      density_particle = prm.get_double ("Density of particle");
      surface_tension = prm.get_double ("Surface tension");
    prm.leave_subsection ();

    prm.enter_subsection ("Droplet Information");
      droplet_radius = prm.get_double ("Droplet radius");
      droplet_real_diameter = prm.get_double ("Droplet real diameter");
      droplet_x_coor = prm.get_double ("X-coord");
      droplet_y_coor = prm.get_double ("Y-coord");
      droplet_z_coor = prm.get_double ("Z-coord"); 
      type_for_thck = prm.get_integer ("Type for thickness");
      fac_thck = prm.get_double ("Factor for thickness");
    prm.leave_subsection ();

    prm.enter_subsection ("Mass Transfer");
      is_set_const_mode = prm.get_bool ("Constant mode inside");
      ratio_soluv = prm.get_double ("Ratio Solubility");
      diffusivity_liquid = prm.get_double ("Diffusivity liquid");
      diffusivity_droplet = prm.get_double ("Diffusivity droplet");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Particle Information");
      type_init_distb = prm.get_double ("Type for initial distribution");
      num_of_par = prm.get_double ("Number of particle");
      par_diameter = prm.get_double ("Diameter");
      factor_visocisty = prm.get_double ("Factor for viscosity");
      is_sedimentaion = prm.get_bool ("Sedimentation");
      dist_from_droplet = prm.get_double ("Distance from droplet");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Period");
      output_fac = prm.get_integer ("output period");
      refine_fac = prm.get_integer ("refine period");
    prm.leave_subsection ();

    prm.enter_subsection ("Problem Definition");
      is_verbal_output = prm.get_bool ("Verbal output");
      error_vel = prm.get_double ("Error NS");
      error_levelset = prm.get_double ("Error level set"); 
      error_concentr = prm.get_double ("Error mass transfer"); 
      error_reinit = prm.get_double ("Error reinit");
      cfl_number = prm.get_double ("CFL number");
      time_step_upl = prm.get_double ("Time interval");
      end_time = prm.get_double ("End time");
      no_buffered_interval = prm.get_integer ("Buffered interval");
    prm.leave_subsection ();
    
    const double grav_acc = 9.8;
    const double den_vis_ratio = density_liquid/viscosity_liquid;
    
    length_ref = droplet_real_diameter;
    velocity_ref = std::sqrt(length_ref*grav_acc);
    time_ref = length_ref/velocity_ref;
    
    Morton_num = (grav_acc*std::pow(viscosity_liquid, 4.0)*density_liquid)/
    (density_liquid*std::pow(surface_tension, 3.0));
    Archimedes_number = den_vis_ratio*std::pow(grav_acc, 0.5)*
   std::pow(droplet_real_diameter, 1.5);    
    Bond_number = (density_liquid*grav_acc*std::pow(droplet_real_diameter, 2.0))
    /surface_tension;
      
    max_level = global_refinement_level + local_refinement_level;
    min_level = global_refinement_level;
    
    avr_droplet_size = 0.0; 
    deformability = 0.0;
    avr_droplet_vel = ( (dim == 2) ? ( Point<dim> (0.0, 0.0)) :
     Point<dim> (0.0, 0.0, 0.0));
    avr_droplet_pos = ( (dim == 2) ? ( Point<dim> (0.0, 0.0)) :
     Point<dim> (0.0, 0.0, 0.0));
    
    pcout << "-- Length Ref = " << length_ref << std::endl;
    pcout << "-- Velocity Ref = " << velocity_ref << std::endl;
    pcout << "-- Time Ref = " << time_ref << std::endl;
    pcout << "-- Morton Number = " << Morton_num << std::endl;
    pcout << "-- Archimedes_number = " << Archimedes_number << std::endl;
    pcout << "-- Bond_number = " << Bond_number << std::endl;
    
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::create_init_coarse_mesh ()
  {
    if (is_verbal_output == true) pcout << "* Create Triangulation.." << std::endl;

    if (iGmsh == false)
    {
      const Point<dim> p1 = ( (dim == 2) ? ( Point<dim> (xmin, ymin)) :
      Point<dim> (xmin, ymin, zmin));
      const Point<dim> p2 = ( (dim == 2) ? ( Point<dim> (xmax, ymax)) :
      Point<dim> (xmax, ymax, zmax));
      
      GridGenerator::hyper_rectangle (triangulation, p1, p2,false);
    }
    else
    {
      GridIn<dim> gmsh_input;
      std::ifstream in(input_mesh_file.c_str());
      gmsh_input.attach_triangulation (triangulation);
      gmsh_input.read_msh (in);
    }

    triangulation.refine_global (global_refinement_level);
    
    for (typename Triangulation<dim>::active_cell_iterator
      cell=triangulation.begin_active();
      cell!=triangulation.end(); ++cell)
    {
      const Point<dim> cell_center = cell->center();

      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
 const Point<dim> face_center = cell->face(f)->center();

 if (cell->face(f)->at_boundary())
 {
   if (std::abs(face_center[0]-xmin) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_x_min);

   if (std::abs(face_center[0]-xmax) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_x_max);

   if (std::abs(face_center[1]-ymin) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_y_min);

   if (std::abs(face_center[1]-ymax) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_y_max);
   
   if (dim == 3)
   {
     if (std::abs(face_center[2]-zmin) < 1e-6)
       cell->face(f)->set_boundary_indicator (bnd_z_min);

     if (std::abs(face_center[2]-zmax) < 1e-6)
       cell->face(f)->set_boundary_indicator (bnd_z_max);
   }
 }
      }
    }
  }

  template <int dim>
  void AdvectionDiffusion<dim>::create_init_adaptive_mesh ()
  {   
    if (local_refinement_level == 0)
      return;
    
    if (is_verbal_output == true) 
      pcout << "* Create Init Adaptive Mesh.." << std::endl;
    
    for (unsigned int i=0; i<local_refinement_level; ++i)
    {
      for (typename Triangulation<dim>::active_cell_iterator
 cell=triangulation.begin_active();
 cell!=triangulation.end(); ++cell)
      {
 const Point<dim> cell_center = cell->center();  
  
 cell->clear_refine_flag ();
 cell->clear_coarsen_flag ();
    
 unsigned int cell_level = cell->level();
 
 // part of droplet
 Point<dim> droplet_pos = ((dim == 2) ? (Point<dim> ( droplet_x_coor,
        droplet_y_coor)) :
        Point<dim> ( droplet_x_coor,
        droplet_y_coor,
        droplet_z_coor));
  
 double xx = std::abs(cell_center.distance(droplet_pos) - droplet_radius);
  
 double aa = 0;
 double bb = 0;
 for (unsigned int i=0; i<local_refinement_level; ++i)
 {
   aa = bb;
   bb = aa + 2*i;
   if (i==0) bb = 1;

   if (xx > aa*safe_fac && xx <= bb*safe_fac)
   {
     if (cell_level < max_level-i) cell->set_refine_flag ();
     if (cell_level > max_level-i) cell->set_coarsen_flag ();
     if (cell_level == max_level-i) {}
   }
 }
  
 if (xx > bb*safe_fac) cell->set_coarsen_flag ();
 
 // part of particle
 Point<dim> c = cell->center();
 std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);
 double min_tq = std::abs (distant_of_par.second);
 if (min_tq < safe_fac*1) cell->set_refine_flag (); 
   
 if (int(cell->level()) == int(min_level))
 cell->clear_coarsen_flag ();
 
      }
      triangulation.execute_coarsening_and_refinement ();
    }
  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::setup_dofs (unsigned int ist)
  {

    GridTools::partition_triangulation (Utilities::Trilinos::get_n_mpi_processes(trilinos_communicator),
      triangulation);

    if (is_verbal_output == true) 
      pcout << "* Setup Dofs.. " << triangulation.n_active_cells() << std::endl;

    //velocity
    {
      dof_handler_velocity.distribute_dofs (fe_velocity);
      DoFRenumbering::subdomain_wise (dof_handler_velocity);
//       DoFRenumbering::Cuthill_McKee (dof_handler_velocity);

      constraint_velocity.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler_velocity,
      constraint_velocity);

      std::set<unsigned char> no_normal_flux_boundaries;
      no_normal_flux_boundaries.insert (5);
      VectorTools::compute_no_normal_flux_constraints (dof_handler_velocity, 0,
       no_normal_flux_boundaries,
       constraint_velocity);
      
      std::set<unsigned char> no_normal_flux_boundaries2;
      no_normal_flux_boundaries2.insert (3);
      VectorTools::compute_no_normal_flux_constraints (dof_handler_velocity, 0,
       no_normal_flux_boundaries2,
       constraint_velocity); 
      constraint_velocity.close ();

      unsigned int n_u = dof_handler_velocity.n_dofs();
      unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_velocity,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
      Epetra_Map map (-1,  local_dofs, 0 , trilinos_communicator);

      vel_star.reinit (map);
      vel_n.reinit (map);
      vel_n_plus_1.reinit (map);
      vel_n_minus_1.reinit (map);
      vel_res.reinit (map);
      rhs_velocity.reinit(map);

      if (ist == 1)
      {
 matrix_velocity.clear();
 TrilinosWrappers::SparsityPattern sp (map);

 DoFTools::make_sparsity_pattern (dof_handler_velocity, sp,
     constraint_velocity, false,
     Utilities::Trilinos::
     get_this_mpi_process(trilinos_communicator));
 sp.compress();
 matrix_velocity.reinit (sp);
      }
    } //End-Velocity

    pcout << "Vel.." << std::endl;
    //pressure
    {
      dof_handler_pressure.distribute_dofs (fe_pressure);
      DoFRenumbering::subdomain_wise (dof_handler_pressure);
      DoFRenumbering::Cuthill_McKee (dof_handler_pressure);

      constraint_pressure.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler_pressure,
      constraint_pressure);
      constraint_pressure.close ();

      unsigned int n_p = dof_handler_pressure.n_dofs();
      unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_pressure,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
      Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

      aux_n_plus_1.reinit (map);
      aux_n.reinit (map);
      aux_n_minus_1.reinit (map);

      pre_n_plus_1.reinit (map);
      pre_n_minus_1.reinit (map);
      pre_star.reinit (map);
      pre_n.reinit (map);
      
      pre_res.reinit (map);
      trn_concentr_on_pressure.reinit (map);

      rhs_pressure.reinit(map);

      if (ist == 1)
      {
 matrix_pressure.clear();
 TrilinosWrappers::SparsityPattern sp (map);

 DoFTools::make_sparsity_pattern (dof_handler_pressure, sp,
       constraint_pressure, false,
       Utilities::Trilinos::
       get_this_mpi_process(trilinos_communicator));
 sp.compress();
 matrix_pressure.reinit (sp);
      }
    } //ENd-Pressure

    pcout << "Pre.." << std::endl;
    
    //levelset
    {
      dof_handler_levelset.distribute_dofs (fe_levelset);
      DoFRenumbering::subdomain_wise (dof_handler_levelset);
      DoFRenumbering::Cuthill_McKee (dof_handler_levelset);

      constraint_levelset.clear ();
      DoFTools::make_hanging_node_constraints ( dof_handler_levelset,
      constraint_levelset);
      constraint_levelset.close ();

      unsigned int n_i = dof_handler_levelset.n_dofs();
      unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

      Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

      rhs_levelset.reinit (map); 
      level_set_curvature.reinit (map);
      levelset_solution.reinit (map);
      old_levelset_solution.reinit (map);
      level_set_grad_x.reinit (map);
      level_set_grad_y.reinit (map);
      level_set_grad_z.reinit (map);       
      level_set_normal_x.reinit (map);
      level_set_normal_y.reinit (map);
      level_set_normal_z.reinit (map);
      particle_at_levelset.reinit (map);
      dimless_density_distribution.reinit (map);
      dimless_viscosity_distribution.reinit (map);

      if (ist == 1)
      {
 matrix_levelset.clear();
 TrilinosWrappers::SparsityPattern sp (map);

 DoFTools::make_sparsity_pattern (dof_handler_levelset, sp,
       constraint_levelset, false,
       Utilities::Trilinos::
       get_this_mpi_process(trilinos_communicator));
 sp.compress();
 matrix_levelset.reinit (sp);
      }
    }//End-Levelset
    
    pcout << "Level.." << std::endl;
    
    // Mass-Transfer
    {
      dof_handler_concentr.distribute_dofs (fe_concentr);
      DoFRenumbering::subdomain_wise (dof_handler_concentr);
      DoFRenumbering::Cuthill_McKee (dof_handler_concentr);

      constraint_concentr.clear ();
      DoFTools::make_hanging_node_constraints ( dof_handler_concentr,
      constraint_concentr);
      constraint_concentr.close ();

      unsigned int n_i = dof_handler_concentr.n_dofs();
      unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

      Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

      concentr_solution.reinit (map);
      old_concentr_solution.reinit (map);
      org_concentration.reinit (map);
      interface_on_cell.reinit (map);
      normal_vec_out_flux_concentr.reinit (map);
      normal_vec_out_flux_concentr2.reinit (map);
      
      rhs_concentr.reinit (map);
      
      if (ist == 1)
      {
 matrix_concentr.clear();
 TrilinosWrappers::SparsityPattern sp (map);

 DoFTools::make_sparsity_pattern (dof_handler_concentr, sp,
       constraint_concentr, false,
       Utilities::Trilinos::
       get_this_mpi_process(trilinos_communicator));
 sp.compress();
 matrix_concentr.reinit (sp);
      }      
    }
    pcout << "Mastra.." << std::endl;
  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::prepare_for_levelset ()
  {
    if (is_verbal_output == true) 
      pcout << "* Set for Level.. ";
    
    smallest_h_size = 10000;
    for (typename Triangulation<dim>::active_cell_iterator
      cell=triangulation.begin_active();
      cell!=triangulation.end(); ++cell)
    {
      smallest_h_size = std::min( smallest_h_size, cell->diameter () );
    }
    
    if (type_for_thck == 0) eps_v_levelset = 0.5*std::pow(smallest_h_size, 0.9);
    if (type_for_thck == 1) eps_v_levelset = fac_thck*smallest_h_size; 
    tau_step = 0.5*std::pow(smallest_h_size, 1.1);
    
    double diff = 2.*eps_v_levelset/smallest_h_size;
    if (is_verbal_output == true) 
    pcout << smallest_h_size << " | " << eps_v_levelset << " | " << diff << std::endl;
  }

  template <int dim>
  void AdvectionDiffusion<dim>::initial_Levelset ()
  {
    if (is_verbal_output == true) 
      pcout <<"* Init. Level Set... " << std::endl;
    
    const Point<dim> droplet_position = ( (dim == 2) ? 
       ( Point<dim> (droplet_x_coor, droplet_y_coor)) :
         Point<dim> ( 
         droplet_x_coor, 
         droplet_y_coor,
         droplet_z_coor));

    MappingQ<dim> ff(fe_levelset.get_degree());
    VectorTools::interpolate ( ff,
    dof_handler_levelset,
    ini_levelset_function<dim> ( droplet_position,
             droplet_radius,
             eps_v_levelset),
    levelset_solution);
  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::initial_particle_distribution ()
  {
    if (num_of_par == 0) return;
    
    if (is_verbal_output == true)
      pcout << "* Particle Generation.. " << type_init_distb << std::endl;

    double xlen = std::abs(xmax-xmin);
    double ylen = std::abs(ymax-ymin);
    double zlen = std::abs(zmax-zmin);
    
    Point<dim> droplet_pos = ((dim == 2) ? ( Point<dim> ( droplet_x_coor,
        droplet_y_coor)) :
           Point<dim> ( droplet_x_coor,
        droplet_y_coor,
        droplet_z_coor));
 
    if (type_init_distb == 0)
    {
      double rp = droplet_radius + dist_from_droplet + 0.5*par_diameter;

      for (unsigned int n=0; n < num_of_par;++n)
      {
 double a1 = 360./double(num_of_par);
 double a2 = 0.0 + n*a1;
 double radian = (3.141592*a2)/180.0;

 Point<dim> dummy;
 
 dummy[0] = droplet_pos[0] + rp*std::sin(radian);
 dummy[1] = droplet_pos[1] + rp*std::cos(radian);

 particle_position.push_back (dummy);
 
 if (is_verbal_output == true)
   pcout << particle_position[n] << " " 
  << par_diameter 
  << std::endl;
      }
    }    
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::refine_mesh ()
  {
    if (local_refinement_level == 0) return;
    
    if (is_verbal_output == true)
      pcout <<"  # Refinement.." << std::endl;
   
    levelset_amr_loop (); 
    adp_execute_transfer ();  
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::levelset_amr_loop ()
  {
    //     if (is_verbal_output == true)
          pcout << "    * Level Set Refine.. " << std::endl;

    const QGauss<dim> quadrature_formula(fe_levelset.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();

    const unsigned int     dofs_per_cell = fe_levelset.dofs_per_cell;
    std::vector<unsigned int>   local_dofs_indices (dofs_per_cell);
    std::vector<double>   level_set_values (n_q_points);

    FEValues<dim>     fe_values_levelset (fe_levelset,
																																										quadrature_formula,
																																										update_values    |
																																										update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    for (; cell!=endc; ++cell)
//     if (cell->subdomain_id() 
//       == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
						cell->clear_coarsen_flag ();
						cell->clear_refine_flag ();
						cell->get_dof_indices (local_dofs_indices);
						unsigned int cell_level = cell->level();

						fe_values_levelset.reinit (cell);

						int case_levelset = 0;
						int case_particle = 0;

						//Part of Level Set
						MappingQ<dim> ff(fe_levelset.get_degree());
						double cc = VectorTools::point_value (ff,
																																												dof_handler_levelset,
																																												levelset_solution,
																																												cell->center());
						cc = std::abs(cc - 0.5);

						double bb_t = 0.5 - std::pow(0.1, drop_adp_fac);
						if (cc <  bb_t) case_levelset = +1;
						if (cc >= bb_t) case_levelset = -1;

						//Part of Particle
						Point<dim> c = cell->center();
						std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);
						double min_tq = std::abs (distant_of_par.second);

						double aa = 0;
						double bb = 0;
						for (unsigned int i=0; i<local_refinement_level; ++i)
						{
								aa = bb;
								bb = aa + 2*i;
								if (i==0) bb = 1;

								if (min_tq > aa*safe_fac && min_tq <= bb*safe_fac)
								{
										if (cell_level < max_level-i) case_particle = +1;
										if (cell_level > max_level-i) case_particle = -1;
								}
						}

						if (cc > 0.5) case_particle = +1;
//							if (min_tq < safe_fac*1) cell->set_refine_flag ();

						if (num_of_par > 0)
						{
								if (case_levelset == +1 || case_particle == +1)
										cell->set_refine_flag ();

								if (case_levelset == -1 && case_particle == -1)
										cell->set_coarsen_flag ();
						}
						else if (num_of_par == 0)
						{
								if (case_levelset == +1) cell->set_refine_flag ();
								if (case_levelset == -1) cell->set_coarsen_flag ();
						}

						if (int(cell->level()) == int(min_level))
								cell->clear_coarsen_flag ();

						if (int(cell->level()) == int(max_level))
								cell->clear_refine_flag ();
		  }
  }
    
  template <int dim>
  void AdvectionDiffusion<dim>::adp_execute_transfer ()
  {
//     if (is_verbal_output == true) 
      pcout << "    * Adp. Exeu. Trans.. " << std::endl;

    std::vector<TrilinosWrappers::Vector> x_vel (4);
    std::vector<TrilinosWrappers::Vector> x_pre (4);
    std::vector<TrilinosWrappers::Vector> x_aux (3);
    std::vector<TrilinosWrappers::Vector> x_levelset (2);
    std::vector<TrilinosWrappers::Vector> x_concentr (2);

    x_vel [0] = vel_n_plus_1;
    x_vel [1] = vel_n;
    x_vel [2] = vel_n_minus_1;
    x_vel [3] = vel_star;

    x_pre [0] = pre_n_plus_1;
    x_pre [1] = pre_n;
    x_pre [2] = pre_n_minus_1;
    x_pre [3] = pre_star;

    x_aux [0] = aux_n_plus_1;
    x_aux [1] = aux_n;
    x_aux [2] = aux_n_minus_1;

    x_levelset [0] = levelset_solution;
    x_levelset [1] = old_levelset_solution;
    
    x_concentr [0] = concentr_solution;
    x_concentr [1] = old_concentr_solution;

    SolutionTransfer<dim,TrilinosWrappers::Vector> vel_trans (dof_handler_velocity);
    SolutionTransfer<dim,TrilinosWrappers::Vector> pre_trans (dof_handler_pressure);
    SolutionTransfer<dim,TrilinosWrappers::Vector> aux_trans (dof_handler_pressure);
    SolutionTransfer<dim,TrilinosWrappers::Vector> level_trans (dof_handler_levelset);
    SolutionTransfer<dim,TrilinosWrappers::Vector> concentr_trans (dof_handler_concentr);
    
    triangulation.prepare_coarsening_and_refinement();

    vel_trans.prepare_for_coarsening_and_refinement(x_vel);
    pre_trans.prepare_for_coarsening_and_refinement(x_pre);
    aux_trans.prepare_for_coarsening_and_refinement(x_aux);
    level_trans.prepare_for_coarsening_and_refinement(x_levelset);
    concentr_trans.prepare_for_coarsening_and_refinement(x_concentr);

    triangulation.execute_coarsening_and_refinement ();

    setup_dofs (1);

    std::vector<TrilinosWrappers::Vector> tmp_vel (4);
    std::vector<TrilinosWrappers::Vector> tmp_pre (4);
    std::vector<TrilinosWrappers::Vector> tmp_aux (3);
    std::vector<TrilinosWrappers::Vector> tmp_leveset (2);
    std::vector<TrilinosWrappers::Vector> tmp_concentr (2);
    
    tmp_vel[0].reinit (vel_n_plus_1);
    tmp_vel[1].reinit (vel_n);
    tmp_vel[2].reinit (vel_n_minus_1);
    tmp_vel[3].reinit (vel_star);

    tmp_pre[0].reinit (pre_n_plus_1);
    tmp_pre[1].reinit (pre_n);
    tmp_pre[2].reinit (pre_n_minus_1);
    tmp_pre[3].reinit (pre_star);

    tmp_aux[0].reinit (aux_n_plus_1);
    tmp_aux[1].reinit (aux_n);
    tmp_aux[2].reinit (aux_n_minus_1);

    tmp_leveset[0].reinit (levelset_solution);
    tmp_leveset[1].reinit (old_levelset_solution);

    tmp_concentr[0].reinit (concentr_solution);
    tmp_concentr[1].reinit (old_concentr_solution);
    
    vel_trans.interpolate (x_vel, tmp_vel);
    pre_trans.interpolate (x_pre, tmp_pre);
    aux_trans.interpolate (x_aux, tmp_aux);
    level_trans.interpolate (x_levelset, tmp_leveset);
    concentr_trans.interpolate (x_concentr, tmp_concentr);
    
    vel_n_plus_1 = tmp_vel[0];
    vel_n = tmp_vel[1];
    vel_n_minus_1 = tmp_vel[2];
    vel_star = tmp_vel[3];

    pre_n_plus_1 = tmp_pre[0];
    pre_n = tmp_pre[1];
    pre_n_minus_1 = tmp_pre[2];
    pre_star = tmp_pre[3];

    aux_n_plus_1 = tmp_aux[0];
    aux_n = tmp_aux[1];
    aux_n_minus_1 = tmp_aux[2];

    levelset_solution = tmp_leveset[0];
    old_levelset_solution = tmp_leveset[1];
    
    concentr_solution = tmp_concentr[0];
    old_concentr_solution = tmp_concentr[1];
    
    pcout << "DONE.." << std::endl;
    
  }

  template <int dim>
  void AdvectionDiffusion<dim>::level_set_compute_normal (unsigned int comp)
  {
    matrix_levelset = 0;
    rhs_levelset = 0;

    QGauss<dim>   quadrature_formula (fe_levelset.get_degree()+1);

    FEValues<dim>  fe_values_levelset (fe_levelset, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);

    const unsigned int   dofs_per_cell = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<double>         phi_T (dofs_per_cell);

    std::vector<Tensor<1,dim> > level_grads(n_q_points);
    std::vector<double> level_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values_levelset.reinit (cell);
      cell->get_dof_indices (local_dofs_indices);

      local_matrix = 0;
      local_rhs = 0;

      fe_values_levelset.get_function_gradients ( 
      levelset_solution,
      level_grads);

      fe_values_levelset.get_function_values (levelset_solution,
      level_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 double b = 0.0;
 for (unsigned int d=0; d<dim; ++d)
 b += level_grads[q][d]*level_grads[q][d];
 b = std::sqrt(b);

 for (unsigned int k=0; k<dofs_per_cell; ++k)
   phi_T[k] = fe_values_levelset.shape_value (k, q);

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     local_matrix(i, j) +=  phi_T [i] *
        phi_T [j] *
        fe_values_levelset.JxW(q);;
   }

   if (b > 1e-10)
   local_rhs (i) +=  phi_T [i] *
       (level_grads[q][comp]/b) *
       fe_values_levelset.JxW(q);

 }
      }

      constraint_levelset.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_levelset);

      constraint_levelset.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_levelset);
    }

    unsigned int n_i = dof_handler_levelset.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
     Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distibuted_solution (map);

    matrix_levelset.compress(VectorOperation::add);
    rhs_levelset.compress(VectorOperation::add);

    SolverControl solver_control (matrix_levelset.m(),
      error_levelset*rhs_levelset.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (matrix_levelset);

    cg.solve ( matrix_levelset, distibuted_solution,
  rhs_levelset, preconditioner);

    if (comp == 0){
      level_set_normal_x = distibuted_solution;
      constraint_levelset.distribute (level_set_normal_x);}

    if (comp == 1){
      level_set_normal_y = distibuted_solution;
      constraint_levelset.distribute (level_set_normal_y);}

    if (comp == 2){
      level_set_normal_z = distibuted_solution;
      constraint_levelset.distribute (level_set_normal_z);}
  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::level_set_2nd_adv_step ()
  {
//     if (is_verbal_output == true)
      pcout << "    * Level Set Advection.. ";

    const bool use_bdf2_scheme = (timestep_number != 0);
    rhs_levelset = 0; matrix_levelset = 0;
    double nu_max = 0.0;

    const QGauss<dim> quadrature_formula(fe_levelset.get_degree()+1);
    FEValues<dim>      fe_values_levelset (fe_levelset, quadrature_formula,
         update_values    |
         update_gradients |
         update_hessians  |
         update_quadrature_points  |
         update_JxW_values);

    FEValues<dim> velocity_fe_values (fe_velocity,
         quadrature_formula,
         update_values |
         update_gradients);


    const unsigned int   dofs_per_cell   = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    Vector<double>       local_rhs (dofs_per_cell);
    FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<Tensor<1,dim> >  velocity_values (n_q_points);
    std::vector<Tensor<1,dim> >  old_velocity_values (n_q_points);

    std::vector<double>          levelSet_values (n_q_points);
    std::vector<double>          old_levelSet_values(n_q_points);

    std::vector<Tensor<1,dim> >  levelSet_grads(n_q_points);
    std::vector<Tensor<1,dim> >  old_levelSet_grads(n_q_points);

    std::vector<double>          levelSet_laplacians(n_q_points);
    std::vector<double>          old_levelSet_laplacians(n_q_points);

    std::vector<double>          phi_T      (dofs_per_cell);
    std::vector<Tensor<1,dim> >  grad_phi_T (dofs_per_cell);
     
    const FEValuesExtractors::Vector velocities (0);

    const std::pair<double,double>
      global_i_range = get_extrapolated_levelSet_range();

    const double average_levelset = 0.5 * (global_i_range.first +
        global_i_range.second);

    const double global_entropy_variation =
      get_entropy_variation_levelset (average_levelset);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      local_rhs = 0;
      local_mass_matrix = 0;

      fe_values_levelset.reinit (cell);
      velocity_fe_values.reinit (vel_cell);

      fe_values_levelset.get_function_values (levelset_solution,
      levelSet_values);
      fe_values_levelset.get_function_values (old_levelset_solution,
      old_levelSet_values);
      fe_values_levelset.get_function_gradients (levelset_solution,
        levelSet_grads);
      fe_values_levelset.get_function_gradients (old_levelset_solution,
        old_levelSet_grads);
      fe_values_levelset.get_function_laplacians (levelset_solution,
          levelSet_laplacians);
      fe_values_levelset.get_function_laplacians (old_levelset_solution,
        old_levelSet_laplacians);
      velocity_fe_values[velocities].get_function_values (vel_n_plus_1,
           velocity_values);
      velocity_fe_values[velocities].get_function_values (vel_n,
           old_velocity_values);

      double nu = 0.0;
      nu
 = compute_viscosity( levelSet_values,
    old_levelSet_values,
    levelSet_grads,
    old_levelSet_grads,
    levelSet_laplacians,
    old_levelSet_laplacians,
    velocity_values,
    old_velocity_values,
    maximal_velocity,
    global_i_range.second - global_i_range.first,
    0.5*(global_i_range.second + global_i_range.first),
    global_entropy_variation,
    cell->diameter());
      nu_max = std::max (nu_max, nu);

      double coef_a1 = 1.0;
      double coef_a2 = time_step_upl;

      if (use_bdf2_scheme == true)
 coef_a1 = (2*time_step_upl + old_time_step_upl) /
    (time_step_upl + old_time_step_upl);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   grad_phi_T[k] = fe_values_levelset.shape_grad (k,q);
   phi_T[k]      = fe_values_levelset.shape_value (k, q);
 }

 const double Ts = (use_bdf2_scheme ?
     (levelSet_values[q] *
     (time_step_upl + old_time_step_upl) / old_time_step_upl
     -
     old_levelSet_values[q] *
     (time_step_upl * time_step_upl) /
     (old_time_step_upl * (time_step_upl + old_time_step_upl)))
     :
     levelSet_values[q]);

 const Tensor<1,dim> ext_grad_T
   =   (use_bdf2_scheme ?
       (levelSet_grads[q] *
       (1+time_step_upl/old_time_step_upl)
       -
       old_levelSet_grads[q] *
       time_step_upl / old_time_step_upl)
       :
       levelSet_grads[q]);

 const Tensor<1,dim> extrapolated_u
   =   (use_bdf2_scheme ?
       (velocity_values[q] * (1+time_step_upl/old_time_step_upl) -
       old_velocity_values[q] * time_step_upl/old_time_step_upl)
       :
       velocity_values[q]);

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     local_mass_matrix(i,j) += coef_a1 *
           phi_T[i] *
           phi_T[j] *
           fe_values_levelset.JxW(q);
   }

   local_rhs(i) += (Ts * phi_T[i]
       -
      time_step_upl *
      extrapolated_u * ext_grad_T * phi_T[i]
       -
      time_step_upl *
      nu * ext_grad_T * grad_phi_T[i])*
      fe_values_levelset.JxW(q);

 }
      }

      cell->get_dof_indices (local_dofs_indices);  

      constraint_levelset.distribute_local_to_global (local_mass_matrix,
       local_dofs_indices,
       matrix_levelset);

      constraint_levelset.distribute_local_to_global ( local_rhs,
        local_dofs_indices,
        rhs_levelset);
    }

    unsigned int n_i = dof_handler_levelset.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distibuted_levelSet (map);

    matrix_levelset.compress(VectorOperation::add);
    rhs_levelset.compress(VectorOperation::add);

    SolverControl solver_control (matrix_levelset.m(),
      error_levelset*rhs_levelset.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (matrix_levelset);

    cg.solve ( matrix_levelset, distibuted_levelSet,
  rhs_levelset, preconditioner);
    levelset_solution = distibuted_levelSet;
    constraint_levelset.distribute (levelset_solution);
    
//     if (is_verbal_output == true)
    pcout << solver_control.last_step()
  << std::endl;
    
    double min_value = levelset_solution(0);
    double max_value = levelset_solution(0);

    for (unsigned int i=0; i<levelset_solution.size(); ++i)
    {
      min_value = std::min<double> (min_value, levelset_solution(i));
      max_value = std::max<double> (max_value, levelset_solution(i));
    }
  }
  
  template <int dim>
  double
  AdvectionDiffusion<dim>::get_entropy_variation_levelset (const double average_levelset) const
  {
    const QGauss<dim> quadrature_formula (fe_levelset.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_levelset, quadrature_formula,
         update_values | update_JxW_values);
    std::vector<double> levelset_values(n_q_points);
    std::vector<double> old_levelset_values(n_q_points);

    double min_entropy = std::numeric_limits<double>::max(),
    max_entropy = -std::numeric_limits<double>::max(),
    area = 0,
    entropy_integrated = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();
     
    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() 
      == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values.reinit (cell);
      fe_values.get_function_values (levelset_solution,
          levelset_values);
      fe_values.get_function_values (old_levelset_solution,
          old_levelset_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 const double T = (levelset_values[q] +
     old_levelset_values[q]) / 2;
 const double entropy = ((T-average_levelset) *
    (T-average_levelset));

 min_entropy = std::min (min_entropy, entropy);
 max_entropy = std::max (max_entropy, entropy);
 area += fe_values.JxW(q);
 entropy_integrated += fe_values.JxW(q) * entropy;
      }
    }

    const double local_sums[2]   = { entropy_integrated, area },
    local_maxima[2] = { -min_entropy, max_entropy };
    double global_sums[2], global_maxima[2];

    Utilities::MPI::sum (local_sums,   MPI_COMM_WORLD, global_sums);
    Utilities::MPI::max (local_maxima, MPI_COMM_WORLD, global_maxima);

    const double average_entropy = global_sums[0] / global_sums[1];
    const double entropy_diff = std::max(global_maxima[1] - average_entropy,
       average_entropy - (-global_maxima[0]));
    return entropy_diff;
  }

  template <int dim>
  double
  AdvectionDiffusion<dim>::get_entropy_variation_concentr (const double average_concentr) const
  {
    const QGauss<dim> quadrature_formula (fe_concentr.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_concentr, quadrature_formula,
         update_values | update_JxW_values);
    std::vector<double> concentr_values(n_q_points);
    std::vector<double> old_concentr_values(n_q_points);

    double min_entropy = std::numeric_limits<double>::max(),
    max_entropy = -std::numeric_limits<double>::max(),
    area = 0,
    entropy_integrated = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();
     
    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() 
      == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values.reinit (cell);
      fe_values.get_function_values (concentr_solution,
          concentr_values);
      fe_values.get_function_values (old_concentr_solution,
          old_concentr_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 const double T = (concentr_values[q] +
     old_concentr_values[q]) / 2;
 const double entropy = ((T-average_concentr) *
    (T-average_concentr));

 min_entropy = std::min (min_entropy, entropy);
 max_entropy = std::max (max_entropy, entropy);
 area += fe_values.JxW(q);
 entropy_integrated += fe_values.JxW(q) * entropy;
      }
    }

    const double  local_sums[2]   = { entropy_integrated, area },
   local_maxima[2] = { -min_entropy, max_entropy };
    double global_sums[2], global_maxima[2];

    Utilities::MPI::sum (local_sums,   MPI_COMM_WORLD, global_sums);
    Utilities::MPI::max (local_maxima, MPI_COMM_WORLD, global_maxima);

    const double average_entropy = global_sums[0] / global_sums[1];
    const double entropy_diff = std::max(global_maxima[1] - average_entropy,
       average_entropy - (-global_maxima[0]));
    return entropy_diff;
  }
  
  template <int dim>
  double
  AdvectionDiffusion<dim>::
  compute_viscosity(   const std::vector<double>          &levelSet,
   const std::vector<double>           &old_levelSet,
   const std::vector<Tensor<1,dim> >   &levelSet_grads,
   const std::vector<Tensor<1,dim> >   &old_levelSet_grads,
   const std::vector<double>           &levelSet_laplacians,
   const std::vector<double>           &old_levelSet_laplacians,
   const std::vector<Tensor<1,dim> >   &velocity_values,
   const std::vector<Tensor<1,dim> >   &old_velocity_values,
   const double                        global_u_infty,
   const double                        global_i_variation,
   const double    average_levelset,
   const double     global_entropy_variation,
   const double                        cell_diameter)
  {

    if (global_u_infty == 0)
    return 5e-3 * cell_diameter;

    const unsigned int n_q_points = levelSet.size();

    double max_residual = 0;
    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
    {
      const Tensor<1,dim> u = (velocity_values[q] +
    old_velocity_values[q]) / 2.0;
      const double T = (levelSet[q] + old_levelSet[q]) / 2.0;
      const double dT_dt = (levelSet[q] - old_levelSet[q])
       / old_time_step_upl;
      const double u_grad_T = u * (levelSet_grads[q] +
         old_levelSet_grads[q]) / 2.0;
      double residual = std::abs(dT_dt + u_grad_T);
      residual *= std::abs(T - average_levelset);

      max_residual = std::max (residual,        max_residual);
      max_velocity = std::max (std::sqrt (u*u), max_velocity);
    }

    const double max_viscosity = (stabilization_beta *
      max_velocity * cell_diameter);
    if (timestep_number == 0)
      return max_viscosity;
    else
    {
      Assert (old_time_step_upl > 0, ExcInternalError());

      double entropy_viscosity;
      entropy_viscosity = (stabilization_c_R *
     cell_diameter * cell_diameter *
     max_residual /
     global_entropy_variation);

      return std::min (max_viscosity, entropy_viscosity);
    }
  }
 
  template <int dim>
  std::pair<double, double>
  AdvectionDiffusion<dim>::
  compute_entropy_viscosity_for_navier_stokes(
                              const std::vector<Tensor<1,dim> >     &old_velocity,
                              const std::vector<Tensor<1,dim> >     &old_old_velocity,
                              const std::vector<Tensor<2,dim> >     &old_velocity_star_grads,
                              const std::vector<Tensor<1,dim> >     &old_velocity_laplacians,
                              const std::vector<Tensor<1,dim> >     &old_pressure_grads,
                              const Tensor<1,dim>                    &source_vector,
                              const double                           coeff1_for_adv,
                              const double                           coeff2_for_visco,
                              const double                           coeff3_for_source,
                              const double                           cell_diameter) const
  {
    const unsigned int n_q_points = old_velocity.size();

    double avr_min_local_viscosity = 0.0;
    double avr_min_local_residual = 0.0;

    double coeff_arti_viscosity = 0.05;
    double maximum_coeff_arti_viscosity = 0.05;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<1,dim> u = old_velocity[q];

      const Tensor<1,dim> du_dt = coeff1_for_adv*
                                  (old_velocity[q]
                                   - old_old_velocity[q])
                                   /time_step_upl;

      const Tensor<1,dim> u_grad_u = coeff1_for_adv*
                                     u*old_velocity_star_grads[q];

      const Tensor<1,dim> u_viscous = coeff2_for_visco*
                                      old_velocity_laplacians[q];

      const Tensor<1,dim> p_grad = old_pressure_grads[q];

      double residual = du_dt*u + u_grad_u*u
                        + p_grad*u - u_viscous*u
                        - coeff3_for_source*source_vector*u;

      double numer_viscosity = coeff_arti_viscosity*
                               cell_diameter*cell_diameter*
                               (std::abs(residual)/(u*u));

      double max_bound_viscosity = maximum_coeff_arti_viscosity*u.norm()*cell_diameter;

      double min_local_viscosity = std::min (max_bound_viscosity, numer_viscosity);

      avr_min_local_viscosity += min_local_viscosity;

      avr_min_local_residual  += residual;
    }

    return std::make_pair (avr_min_local_residual/double(n_q_points),
                           avr_min_local_viscosity/double(n_q_points));
  }
  template <int dim>
  std::pair<double,double>
  AdvectionDiffusion<dim>::get_extrapolated_levelSet_range ()
  {
    double min_levelset, max_levelset;

    const QIterated<dim> quadrature_formula (QTrapez<1>(), fe_levelset.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_levelset, quadrature_formula,
         update_values);
    std::vector<double> levelset_values(n_q_points);
    std::vector<double> old_levelset_values(n_q_points);

    if (timestep_number != 0)
    {
      min_levelset = (1. + time_step_upl/old_time_step_upl) *
        levelset_solution.linfty_norm()
        +
        time_step_upl/old_time_step_upl *
        old_levelset_solution.linfty_norm(),
      max_levelset = -min_levelset;

      typename DoFHandler<dim>::active_cell_iterator
 cell = dof_handler_levelset.begin_active(),
 endc = dof_handler_levelset.end();
      for (; cell!=endc; ++cell)
      {
 fe_values.reinit (cell);
 fe_values.get_function_values (levelset_solution,
     levelset_values);
 fe_values.get_function_values (old_levelset_solution,
     old_levelset_values);
 for (unsigned int q=0; q<n_q_points; ++q)
 {
   const double levelset =
   (1. + time_step_upl/old_time_step_upl) * levelset_values[q]-
   time_step_upl/old_time_step_upl * old_levelset_values[q];

   min_levelset = std::min (min_levelset, levelset);
   max_levelset = std::max (max_levelset, levelset);
 }
      }
      return std::make_pair(min_levelset, max_levelset);
    }
    else
    {
      min_levelset = levelset_solution.linfty_norm(),
      max_levelset = -min_levelset;

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();
      for (; cell!=endc; ++cell)
      {
 fe_values.reinit (cell);
 fe_values.get_function_values (levelset_solution,
     levelset_values);

 for (unsigned int q=0; q<n_q_points; ++q)
 {
   const double levelset = levelset_values[q];

   min_levelset = std::min (min_levelset, levelset);
   max_levelset = std::max (max_levelset, levelset);
 }
      }
      return std::make_pair(min_levelset, max_levelset);
    }
  }

  template <int dim>
  std::pair<double,double>
  AdvectionDiffusion<dim>::get_extrapolated_concentr_range ()
  {
    double min_concentr, max_concentr;

    const QIterated<dim> quadrature_formula (QTrapez<1>(), fe_concentr.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_concentr, quadrature_formula,
         update_values);
    std::vector<double> concentr_values(n_q_points);
    std::vector<double> old_concentr_values(n_q_points);

    if (timestep_number != 0)
    {
      min_concentr = (1. + time_step_upl/old_time_step_upl) *
        concentr_solution.linfty_norm()
        +
        time_step_upl/old_time_step_upl *
        old_concentr_solution.linfty_norm(),
      max_concentr = -min_concentr;

      typename DoFHandler<dim>::active_cell_iterator
 cell = dof_handler_concentr.begin_active(),
 endc = dof_handler_concentr.end();
      for (; cell!=endc; ++cell)
      {
 fe_values.reinit (cell);
 fe_values.get_function_values (concentr_solution,
     concentr_values);
 fe_values.get_function_values (old_concentr_solution,
     old_concentr_values);
 for (unsigned int q=0; q<n_q_points; ++q)
 {
   const double concentr =
   (1. + time_step_upl/old_time_step_upl) * concentr_values[q]-
   time_step_upl/old_time_step_upl * old_concentr_values[q];

   min_concentr = std::min (min_concentr, concentr);
   max_concentr = std::max (max_concentr, concentr);
 }
      }
      return std::make_pair(min_concentr, max_concentr);
    }
    else
    {
      min_concentr = concentr_solution.linfty_norm(),
      max_concentr = -min_concentr;

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();
      for (; cell!=endc; ++cell)
      {
 fe_values.reinit (cell);
 fe_values.get_function_values (concentr_solution,
     concentr_values);

 for (unsigned int q=0; q<n_q_points; ++q)
 {
   const double concentr = concentr_values[q];

   min_concentr = std::min (min_concentr, concentr);
   max_concentr = std::max (max_concentr, concentr);
 }
      }
      return std::make_pair(min_concentr, max_concentr);
    }
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::level_set_reinitial_step ()
  {
    matrix_levelset = 0;
    rhs_levelset = 0;
     
    QGauss<dim>    quadrature_formula (fe_levelset.get_degree()+1);
    FEValues<dim>   fe_values_levelset (fe_levelset, quadrature_formula,
         update_values    |
         update_quadrature_points  |
         update_JxW_values |
         update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity,
         quadrature_formula,
         update_values);

    const unsigned int   dofs_per_cell   = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<double>         phi_T       (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T  (dofs_per_cell);

    std::vector<Tensor<1,dim> > level_grads(n_q_points);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);
    std::vector<double> level_values (n_q_points);
     
    std::vector<double> level_nrm_x (n_q_points);
    std::vector<double> level_nrm_y (n_q_points);
    std::vector<double> level_nrm_z (n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      local_matrix = 0;
      local_rhs = 0;

      fe_values_levelset.reinit (cell);
      fe_values_velocity.reinit (vel_cell);

      fe_values_levelset.get_function_gradients (levelset_solution, level_grads);
      fe_values_levelset.get_function_values (levelset_solution, level_values);
      fe_values_levelset.get_function_values (level_set_normal_x, level_nrm_x);
      fe_values_levelset.get_function_values (level_set_normal_y, level_nrm_y);
      fe_values_levelset.get_function_values (level_set_normal_z, level_nrm_z);
      fe_values_velocity[velocities].get_function_values (vel_n_plus_1, velocity_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {

 Point<dim> nrm_val;
 nrm_val [0] = level_nrm_x [q];
 nrm_val [1] = level_nrm_y [q];
 if (dim == 3) nrm_val [2] = level_nrm_z [q];

 for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   grad_phi_T[k] = fe_values_levelset.shape_grad (k, q);
   phi_T[k]      = fe_values_levelset.shape_value (k, q);
 }

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     // time adv
     local_matrix (i,j) += phi_T [i]*
     phi_T [j]*
     fe_values_levelset.JxW(q);

     local_matrix (i, j) -=   tau_step *
     0.5 *
     grad_phi_T [i] *
     phi_T [j] *
     nrm_val *
     fe_values_levelset.JxW(q);

     local_matrix (i, j) +=   tau_step *
     grad_phi_T [i] *
     phi_T [j] *
     nrm_val *
     level_values [q] *
     fe_values_levelset.JxW(q);

     // diffusion term
//  local_matrix (i, j) +=  tau_step *
//       eps_v_levelset * 0.5 *
//       grad_phi_T [i] *
//       nrm_val *
//       grad_phi_T [j] *
//       nrm_val *
//       fe_values_levelset.JxW(q);
        
     local_matrix (i, j) +=    tau_step *
     eps_v_levelset * 
     grad_phi_T [i] *
     grad_phi_T [j] *
     fe_values_levelset.JxW(q);
   }

   // time adv
   local_rhs (i) += phi_T [i] *
    level_values[q] *
    fe_values_levelset.JxW(q);

   local_rhs (i) += tau_step *
    0.5 *
    grad_phi_T [i] *
    nrm_val *
    level_values [q] *
    fe_values_levelset.JxW(q);

   // diffusion term
//   local_rhs (i) -= inv_mu*
//      tau_step *
//      eps_v_levelset * 0.5 *
//      level_grads [q] *
//      nrm_val *
//      grad_phi_T [i] *
//      nrm_val *
//      fe_values_levelset.JxW(q);
 }
      }

      cell->get_dof_indices (local_dofs_indices);

      constraint_levelset.distribute_local_to_global ( local_matrix,
       local_dofs_indices,
       matrix_levelset);

      constraint_levelset.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_levelset);

    }

    unsigned int n_i = dof_handler_levelset.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_solution (map);
      
    matrix_levelset.compress(VectorOperation::add);
    rhs_levelset.compress(VectorOperation::add);

    SolverControl solver_control (matrix_levelset.m(), error_levelset*rhs_levelset.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector>
    gmres (solver_control, SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData(100));

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_levelset);
    gmres.solve (matrix_levelset, distibuted_solution, rhs_levelset, preconditioner);

    levelset_solution = distibuted_solution;
    constraint_levelset.distribute (levelset_solution);
  }

  template <int dim>
  void AdvectionDiffusion<dim>::sufTen_compute_gradient (unsigned int comp)
  {
  //     if (is_verbal_output == true)
//         pcout << "## sufTen_compute_gradient " << comp << std::endl;

    matrix_levelset = 0;
    rhs_levelset = 0;

    QGauss<dim>   quadrature_formula (fe_levelset.get_degree()+1);
    FEValues<dim>  fe_values_levelset (fe_levelset, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);

    const unsigned int   dofs_per_cell = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<double> phi_T (dofs_per_cell);

    std::vector<Tensor<1,dim> > level_grads(n_q_points);
    std::vector<double> level_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() 
      == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values_levelset.reinit (cell);
      cell->get_dof_indices (local_dofs_indices);

      local_matrix = 0;
      local_rhs = 0;
      
      fe_values_levelset.get_function_gradients (levelset_solution,
        level_grads);

      fe_values_levelset.get_function_values (levelset_solution,
      level_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 double b = 0.0;
 for (unsigned int d=0; d<dim; ++d)
 b += level_grads[q][d]*level_grads[q][d];
 b = std::sqrt(b);

 for (unsigned int k=0; k<dofs_per_cell; ++k)
   phi_T[k]      = fe_values_levelset.shape_value (k, q);

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     local_matrix(i, j) +=  phi_T [i] *
     phi_T [j] *
     fe_values_levelset.JxW(q);
   }

   if (b > 1e-10)
   local_rhs (i) +=  phi_T [i] *
       level_grads[q][comp] *
       fe_values_levelset.JxW(q);

 }
      }

      constraint_levelset.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_levelset);

      constraint_levelset.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_levelset);
    }
  
    unsigned int n_i = dof_handler_levelset.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_solution (map);

    matrix_levelset.compress(VectorOperation::add);
    rhs_levelset.compress(VectorOperation::add);

    SolverControl solver_control (matrix_levelset.m(),
      error_levelset*rhs_levelset.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (matrix_levelset);

    cg.solve (matrix_levelset, distibuted_solution,
       rhs_levelset, preconditioner);

    if (comp == 0){
      level_set_grad_x = distibuted_solution;
      constraint_levelset.distribute (level_set_grad_x);}

    if (comp == 1){
      level_set_grad_y = distibuted_solution;
      constraint_levelset.distribute (level_set_grad_y);}

    if (comp == 2 && dim == 3){
      level_set_grad_z = distibuted_solution;
      constraint_levelset.distribute (level_set_grad_z);}
  }

  template <int dim>
  void AdvectionDiffusion<dim>::surTen_compute_curvature ()
  {
  //     if (is_verbal_output == true)
//         pcout << "## surTen_compute_curvature " << std::endl;

    matrix_levelset = 0;
    rhs_levelset = 0;

    QGauss<dim>    quadrature_formula (fe_levelset.get_degree()+1);
    FEValues<dim>  fe_values_levelset (fe_levelset, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);

    const unsigned int   dofs_per_cell = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    
    std::vector<double>         phi_T       (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);
    std::vector<Tensor<1,dim> > level_set_grads (n_q_points);
    std::vector<double> level_set_values (n_q_points);

    std::vector<double> level_grd_x (n_q_points);
    std::vector<double> level_grd_y (n_q_points);
    std::vector<double> level_grd_z (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_levelset.begin_active(),
      endc = dof_handler_levelset.end();

    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() 
      == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      local_matrix = 0;
      local_rhs = 0;

      fe_values_levelset.reinit (cell);
      fe_values_levelset.get_function_gradients (levelset_solution, level_set_grads);
      fe_values_levelset.get_function_values (levelset_solution, level_set_values);
      fe_values_levelset.get_function_values (level_set_grad_x, level_grd_x);
      fe_values_levelset.get_function_values (level_set_grad_y, level_grd_y);

      if (dim == 3)
 fe_values_levelset.get_function_values (level_set_grad_z, level_grd_z);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 Point<dim> nor_val;
 nor_val [0] = level_grd_x [q];
 nor_val [1] = level_grd_y [q];
 if (dim == 3) nor_val [2] = level_grd_z [q];

 double b = 0.0;
 for (unsigned int d=0; d<dim; ++d)
   b += nor_val[d]*nor_val[d];
 b = std::sqrt(b);
     
 nor_val = nor_val/b;
   
 for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   grad_phi_T[k] = fe_values_levelset.shape_grad (k, q);
   phi_T[k]      = fe_values_levelset.shape_value (k, q);
 }

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     local_matrix(i,j) += phi_T[i]*
      phi_T[j]*
      fe_values_levelset.JxW(q);

   }

   if (b > 1e-10)
   local_rhs (i) +=  grad_phi_T[i]*
    nor_val*
    fe_values_levelset.JxW(q);
    

 }
      }

      cell->get_dof_indices (local_dofs_indices);

      constraint_levelset.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_levelset);

      constraint_levelset.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_levelset);

      }

      unsigned int n_i = dof_handler_levelset.n_dofs();
      unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
      Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
      TrilinosWrappers::MPI::Vector distibuted_solution (map);

      matrix_levelset.compress(VectorOperation::add);
      rhs_levelset.compress(VectorOperation::add);

      SolverControl solver_control (matrix_levelset.m(), error_levelset*rhs_levelset.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

      TrilinosWrappers::PreconditionIC preconditioner;
      preconditioner.initialize (matrix_levelset);

      cg.solve (matrix_levelset, distibuted_solution, rhs_levelset, preconditioner);

      level_set_curvature = distibuted_solution;
      constraint_pressure.distribute (level_set_curvature);

  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::diffusion_step ()
  {
    // if (is_verbal_output == true) 
    pcout << "    * Diffusion Step..  ";

    matrix_velocity = 0;
    rhs_velocity = 0;

    double inv_time_step_upl = 1./time_step_upl;

    const QGauss<dim> quadrature_formula((fe_pressure.get_degree() + 1)+1);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
          update_values    |
          update_quadrature_points  |
          update_JxW_values |
          update_gradients);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
          update_values    |
          update_quadrature_points  |
          update_JxW_values |
          update_gradients);

    FEValues<dim> fe_values_levelset (fe_levelset, quadrature_formula,
          update_values    |
          update_quadrature_points  |
          update_JxW_values |
          update_gradients);

    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    std::vector<unsigned int> local_con_dof_indices_pre (fe_pressure.dofs_per_cell);
    std::vector<unsigned int> local_dof_indices_levelset (fe_levelset.dofs_per_cell);

    std::vector<double>  pre_star_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_star_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_n_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_n_minus_1_values (n_q_points);
    std::vector<Tensor<2,dim> > grad_vel_star_values (n_q_points);

    std::vector<Tensor<1,dim> > grad_aux_n_values (n_q_points);
    std::vector<Tensor<1,dim> > grad_aux_n_minus_1_values (n_q_points);

    std::vector<Tensor<1,dim> > grad_pre_n_values (n_q_points);

    std::vector<double> level_set_values (n_q_points);
    std::vector<double> level_curvature_values (n_q_points);
    std::vector<double> level_grd_x (n_q_points);
    std::vector<double> level_grd_y (n_q_points);
    std::vector<double> level_grd_z (n_q_points);
    std::vector<Tensor<1,dim> > level_set_grads (n_q_points);

    std::vector<double> dimless_density_values (n_q_points);
    std::vector<double> dimless_viscosity_values (n_q_points);
  
    std::vector<Tensor<1,dim> >          phi_u (dofs_per_cell);
    std::vector<Tensor<2,dim> >          grads_phi_u (dofs_per_cell);
    std::vector<SymmetricTensor<2,dim> > symm_grads_phi_u (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    typename DoFHandler<dim>::active_cell_iterator
      pre_cell = dof_handler_pressure.begin_active();

    typename DoFHandler<dim>::active_cell_iterator
      level_cell = dof_handler_levelset.begin_active();

    for (; cell!=endc; ++cell, ++pre_cell, ++level_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values_velocity.reinit (cell);
      fe_values_pressure.reinit (pre_cell);
      fe_values_levelset.reinit (level_cell);

      cell->get_dof_indices (local_dofs_indices);
      pre_cell->get_dof_indices (local_con_dof_indices_pre);
      level_cell->get_dof_indices (local_dof_indices_levelset);

      fe_values_velocity[velocities].get_function_values (vel_star, vel_star_values);
      fe_values_velocity[velocities].get_function_values (vel_n_minus_1, vel_n_minus_1_values);
      fe_values_velocity[velocities].get_function_values (vel_n, vel_n_values);
      fe_values_velocity[velocities].get_function_gradients (vel_star, grad_vel_star_values);

      fe_values_pressure.get_function_gradients (aux_n, grad_aux_n_values);
      fe_values_pressure.get_function_gradients (aux_n_minus_1, grad_aux_n_minus_1_values);
      fe_values_pressure.get_function_values (pre_star, pre_star_values);
      fe_values_pressure.get_function_gradients (pre_n, grad_pre_n_values);

      fe_values_levelset.get_function_values (levelset_solution, level_set_values);
      fe_values_levelset.get_function_values (level_set_curvature, level_curvature_values);
      fe_values_levelset.get_function_values (level_set_grad_x, level_grd_x);
      fe_values_levelset.get_function_values (level_set_grad_y, level_grd_y);
      fe_values_levelset.get_function_values (level_set_grad_z, level_grd_z);

      fe_values_levelset.get_function_values (dimless_density_distribution, dimless_density_values);
      fe_values_levelset.get_function_values (dimless_viscosity_distribution, dimless_viscosity_values);

      local_matrix = 0;
      local_rhs = 0;
      
      double dimless_density = 1.0;
      double dimless_viscosity = 1.0;

      double cc = 0.0;

      {
 MappingQ<dim> ff (fe_levelset.get_degree());
 cc = VectorTools::point_value (ff,
     dof_handler_levelset,
     levelset_solution,
     cell->center());
 if (cc<0.0) cc = 0.0;
 if (cc>1.0) cc = 1.0;

 double density_ref = density_liquid;
 double ratio_density_drp = density_droplet/density_ref;
 double ratio_density_liq = density_liquid/density_ref;
 dimless_density = cc*ratio_density_drp + (1.0-cc)*ratio_density_liq;

 double viscosity_ref = viscosity_liquid;
 double ratio_viscosity_drp = viscosity_droplet/viscosity_ref;
 double ratio_viscosity_liq = viscosity_liquid/viscosity_ref;
 dimless_viscosity = cc*ratio_viscosity_drp + (1.0-cc)*ratio_viscosity_liq;
 
    Tensor <1, dim> source_vector;
//    for (unsigned int d=0; d<dim; ++d)
//      source_vector[d] = parameters.inclined_angle_vector[d];
//    double coeff1_for_adv = coeff_with_adv_term;
//    double coeff2_for_visco =  current_q_viscosity/parameters.Reynolds_number;
//    double coeff3_for_source = (theta/(parameters.Froude_number*parameters.Froude_number));
    std::pair<double, double> entropy_pair;

//  if (timestep_number > 2)
//    {
//      entropy_pair =
//        compute_entropy_viscosity_for_navier_stokes(vel_n_values,
//                                                    vel_n_minus_1_values,
//                                                    grad_vel_star_values,
//                                                    laplacian_vel_star_values,
//                                                    grad_pre_n_values,
//                                                    source_vector,
//                                                    coeff1_for_adv,
//                                                    coeff2_for_visco,
//                                                    coeff3_for_source,
//                                                    cell->diameter());
//    }

 if (num_of_par > 0)
 {
   Point<dim> c = cell->center();
   std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);
   
   if (distant_of_par.second < 0.0)
   {
     dimless_density = density_particle/density_ref;
     dimless_viscosity = viscosity_liquid*factor_visocisty/viscosity_ref;
   }
 }
   
      }
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   phi_u[k] = fe_values_velocity[velocities].value (k,q);
   grads_phi_u[k] = fe_values_velocity[velocities].gradient (k,q);
   symm_grads_phi_u[k] = fe_values_velocity[velocities].symmetric_gradient(k,q);
 }

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   //Time-stepping
   local_rhs(i) -=  phi_u[i]*
       (
         -2.0*
         vel_n_values[q]
         +
         0.5*
         vel_n_minus_1_values[q]
       )*
       fe_values_velocity.JxW(q);

   //Gradient of pressure
   local_rhs(i) -= time_step_upl*
     (1./dimless_density)*
     phi_u[i]*
     (
       grad_pre_n_values[q] +
       (4./3.)*grad_aux_n_values[q] -
       (1./3.)*grad_aux_n_minus_1_values[q]
     )*
     fe_values_velocity.JxW(q);

   // body-force term : bouyant and surface tension
   {
     const Point<dim> gravity = ( (dim == 2) ? (Point<dim> (0,-1)) :
     Point<dim> (0,0,-1));

     Point<dim> grad_xyz;
     grad_xyz [0] = level_grd_x [q];
     grad_xyz [1] = level_grd_y [q];
     if (dim == 3) grad_xyz [2] = level_grd_z [q];

     local_rhs(i) +=  time_step_upl*
    ((dimless_density-1.)/
    dimless_density)*
    gravity*
    phi_u[i]*
    fe_values_velocity.JxW(q);

     local_rhs(i) +=  time_step_upl*
    (1./Bond_number)*
    (1./dimless_density)*
    phi_u[i]*
    fe_values_velocity.JxW (q)*
    level_curvature_values [q]*
    grad_xyz;

   }

   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     const unsigned int comp_j = fe_velocity.system_to_component_index (j).first;

     //Time-stepping
     local_matrix(i,j) +=   1.5*
      phi_u[i]*
      phi_u[j]*
      fe_values_velocity.JxW(q);

     //Advection Term
     {
       local_matrix(i,j) += time_step_upl*
     grads_phi_u[j]*
     vel_star_values[q]*
     phi_u[i]*
     fe_values_velocity.JxW(q);

       double bb = 0.0;
       for (unsigned int d=0; d<dim; ++d)
  bb += grad_vel_star_values[q][d][d];
        
       local_matrix(i,j) += time_step_upl*
     0.5*
     phi_u[i]*
     bb*
     phi_u[j]*
     fe_values_velocity.JxW(q);
     }

									//Viscous term
									local_matrix(i,j) += time_step_upl*
																														(dimless_viscosity/
																														dimless_density)*
																														(2.0/Archimedes_number)*
																														symm_grads_phi_u[i]*
																														symm_grads_phi_u[j]*
																														fe_values_velocity.JxW(q);

//         //Artificial Viscous term
//         local_matrix(i,j) += (1./dimless_density)*
//																														time_step_upl*
//																														entropy_pair.second*
//																														symm_grads_phi_u[i]*
//																														symm_grads_phi_u[j]*
//																														fe_values_velocity.JxW(q);

   }
 }
      }
      
      constraint_velocity.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_velocity);

      constraint_velocity.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_velocity);
    }

    std::map<unsigned int,double> boundary_values;
    std::vector<bool> vel_prof(dim, true);

    unsigned int n_u = dof_handler_velocity.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_velocity,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distributed_vel_n_plus_1 (map);

    VectorTools::interpolate_boundary_values (dof_handler_velocity,
      6,
      ZeroFunction<dim>(dim),
      boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
     matrix_velocity,
     distributed_vel_n_plus_1,
     rhs_velocity,
     false);

    matrix_velocity.compress(VectorOperation::add);
    rhs_velocity.compress(VectorOperation::add);

    distributed_vel_n_plus_1 = vel_n_plus_1;
    SolverControl solver_control (matrix_velocity.m(), error_vel*rhs_velocity.l2_norm ());

    SolverGMRES<TrilinosWrappers::MPI::Vector>
    gmres (solver_control,SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData(100));

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_velocity);
    gmres.solve (matrix_velocity, distributed_vel_n_plus_1, rhs_velocity, preconditioner);

    vel_n_plus_1 = distributed_vel_n_plus_1;

//      if (is_verbal_output == true)
    pcout << solver_control.last_step()
  << std::endl;
    constraint_velocity.distribute (vel_n_plus_1);
  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::projection_step()
  {
  //     if (is_verbal_output == true) 
    pcout << "    * Projection Step.. ";

    matrix_pressure = 0;
    rhs_pressure = 0;

    double inv_time_step_upl = 1./time_step_upl;

    const QGauss<dim> quadrature_formula (fe_pressure.get_degree()+1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
          update_values    |
          update_quadrature_points  |
          update_JxW_values |
          update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
          update_values |
          update_gradients);

    FEValues<dim> fe_values_levelset (fe_levelset, quadrature_formula,
          update_values |
          update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices_levelset (dofs_per_cell);
    
    std::vector<Tensor<2,dim> > grad_vel_n_plus_1_values (n_q_points);

    std::vector<double> level_set_values (n_q_points);
    std::vector<double> dimless_density_values (n_q_points);
    std::vector<double> dimless_viscosity_values (n_q_points);
     
    const FEValuesExtractors::Vector velocities (0);
    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_pressure.begin_active(),
      endc = dof_handler_pressure.end();

    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();

    typename DoFHandler<dim>::active_cell_iterator
      level_cell = dof_handler_levelset.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell, ++level_cell) 
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values_pressure.reinit (cell);
      fe_values_velocity.reinit (vel_cell);
      fe_values_levelset.reinit (level_cell);

      fe_values_velocity[velocities].get_function_gradients(vel_n_plus_1, 
      grad_vel_n_plus_1_values);
      fe_values_levelset.get_function_values (levelset_solution, 
      level_set_values);
      fe_values_levelset.get_function_values (dimless_density_distribution, 
      dimless_density_values);
      fe_values_levelset.get_function_values (dimless_viscosity_distribution, 
      dimless_viscosity_values);
      
      cell->get_dof_indices (local_dofs_indices);

      local_matrix = 0;
      local_rhs = 0;
      
      double dimless_density = 1.0;
      double dimless_viscosity = 1.0;

      double cc = 0.0;
      MappingQ<dim> ff (fe_levelset.get_degree());
      cc = VectorTools::point_value (ff,
          dof_handler_levelset,
          levelset_solution,
          cell->center());
      if (cc<0.0) cc = 0.0;
      if (cc>1.0) cc = 1.0;

      double density_ref = density_liquid;
      double ratio_density_drp = density_droplet/density_ref;
      double ratio_density_liq = density_liquid/density_ref;
      dimless_density = cc*ratio_density_drp + (1.0-cc)*ratio_density_liq;

      double viscosity_ref = viscosity_liquid;
      double ratio_viscosity_drp = viscosity_droplet/viscosity_ref;
      double ratio_viscosity_liq = viscosity_liquid/viscosity_ref;
      dimless_viscosity = cc*ratio_viscosity_drp + (1.0-cc)*ratio_viscosity_liq;

      if (num_of_par > 0)
      {
 Point<dim> c = cell->center();
 std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);
   
 if (distant_of_par.second < 0.0)
 {
   dimless_density = density_particle/density_ref;
   dimless_viscosity = viscosity_liquid*factor_visocisty/viscosity_ref;
 }
      }
 
      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int i=0; i<dofs_per_cell; ++i)
 for (unsigned int j=0; j<dofs_per_cell; ++j)
   local_matrix(i,j) -=  (1./dimless_density)*
      time_step_upl*
      0.75*
      fe_values_pressure.shape_grad(i, q)*
      fe_values_pressure.shape_grad(j, q)*
      fe_values_pressure.JxW(q);

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   double bb = 0.0;
   for (unsigned int d=0; d<dim; ++d)
     bb += grad_vel_n_plus_1_values[q][d][d];
         
   local_rhs(i) += fe_values_pressure.shape_value(i, q)*
    bb*
    fe_values_pressure.JxW(q);

 }
      }

      constraint_pressure.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_pressure);

      constraint_pressure.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_pressure);
    }

    std::map<unsigned int,double> boundary_values;

//     typename DoFHandler<dim>::active_cell_iterator  initial_cell = dof_handler_pressure.begin_active();
//     unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , 0);
//     boundary_values[dof_number_pressure] = 0.0;

    MappingQ<dim> ff (fe_pressure.get_degree());
    VectorTools::interpolate_boundary_values (ff,
      dof_handler_pressure,
      3,
      ConstantFunction<dim>(0.0),
      boundary_values);
     
    unsigned int n_p = dof_handler_pressure.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_pressure,
     Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_aux_n_plus_1 (map);

    MatrixTools::apply_boundary_values (boundary_values,
     matrix_pressure,
     distibuted_aux_n_plus_1,
     rhs_pressure,
     false);

    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    distibuted_aux_n_plus_1 = aux_n_plus_1;
    SolverControl solver_control (matrix_pressure.m(), error_vel*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_aux_n_plus_1, rhs_pressure, preconditioner);
    aux_n_plus_1 = distibuted_aux_n_plus_1;

    //     if (is_verbal_output == true)
          pcout << solver_control.last_step()
      << std::endl;

    constraint_pressure.distribute (aux_n_plus_1);
  }

  template <int dim>
  void AdvectionDiffusion<dim>::pressure_correction_step_rot ()
  {
//     if (is_verbal_output == true)
      pcout << "    * Pressure Step.. ";

    matrix_pressure = 0;
    rhs_pressure = 0;

    double inv_time_step_upl = 1./time_step_upl;

    const QGauss<dim> quadrature_formula (fe_pressure.get_degree()+1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
     update_values |
     update_gradients);

    FEValues<dim> fe_values_levelset (fe_levelset, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);
  
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);
    
    std::vector<Tensor<2,dim> > grad_vel_n_plus_1_values (n_q_points);
//     std::vector<Tensor<1,dim>vel_n_plus_1_value (n_q_points);
    
    std::vector<double>         level_set_values (n_q_points);
    std::vector<double>         aux_nPlus_values (n_q_points);
    std::vector<double>         pre_n_values (n_q_points);
     
    std::vector<double>         dimless_density_values (n_q_points);
    std::vector<double>         dimless_viscosity_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_pressure.begin_active(),
      endc = dof_handler_pressure.end();

    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();

    typename DoFHandler<dim>::active_cell_iterator
      level_cell = dof_handler_levelset.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell, ++level_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      cell->get_dof_indices (local_dofs_indices);

      fe_values_pressure.reinit (cell);
      fe_values_velocity.reinit (vel_cell);
      fe_values_levelset.reinit (level_cell);

      fe_values_velocity[velocities].get_function_gradients (vel_n_plus_1, grad_vel_n_plus_1_values);
      fe_values_levelset.get_function_values (levelset_solution, level_set_values);
      fe_values_pressure.get_function_values (aux_n_plus_1, aux_nPlus_values);
      fe_values_pressure.get_function_values (pre_n, pre_n_values);

      fe_values_levelset.get_function_values (dimless_density_distribution, 
      dimless_density_values);
      fe_values_levelset.get_function_values (dimless_viscosity_distribution, 
      dimless_viscosity_values);
      
      local_matrix = 0;
      local_rhs = 0;
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int i=0; i<dofs_per_cell; ++i)
 for (unsigned int j=0; j<dofs_per_cell; ++j)
 {
   local_matrix(i,j) +=  fe_values_pressure.shape_value(i, q)*
        fe_values_pressure.shape_value(j, q)*
        fe_values_pressure.JxW(q);

 }

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   double bb = 0.0;
   for (unsigned int d=0; d<dim; ++d)
     bb += grad_vel_n_plus_1_values [q][d][d];

   local_rhs(i) += fe_values_pressure.shape_value(i, q)*
     (
       pre_n_values[q] +
       aux_nPlus_values[q] 
//      -
//       ((dimless_viscosity_values[q]/dimless_density_values[q])
//       /Archimedes_number)
//       *bb
     )*
     fe_values_pressure.JxW(q);
 }
      }

      constraint_pressure.distribute_local_to_global (local_matrix,
       local_dofs_indices,
       matrix_pressure);

      constraint_pressure.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_pressure);
    }

    unsigned int n_p = dof_handler_pressure.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_pressure,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distibuted_pre_n_plus_1 (map);

    std::map<unsigned int,double> boundary_values;

//      MappingQ<dim> ff (fe_pressure.get_degree());
//      VectorTools::interpolate_boundary_values (ff,
//        dof_handler_pressure,
//        3,
//        ConstantFunction<dim>(100.0),
//        boundary_values);

//     typename DoFHandler<dim>::active_cell_iterator  initial_cell = dof_handler_pressure.begin_active();
//     unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , 0);
//     boundary_values[dof_number_pressure] = 0.0;

    MatrixTools::apply_boundary_values (boundary_values,
     matrix_pressure,
     distibuted_pre_n_plus_1,
     rhs_pressure,
     false);
     
    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    SolverControl solver_control (matrix_pressure.m(), error_vel*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    //TrilinosWrappers::PreconditionILU preconditioner;
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_pre_n_plus_1, rhs_pressure, preconditioner);
    pre_n_plus_1 = distibuted_pre_n_plus_1;

    //     if (is_verbal_output == true)
    pcout << solver_control.last_step()
   << std::endl;

    constraint_pressure.distribute (pre_n_plus_1);
  }

  template <int dim>
  void AdvectionDiffusion<dim>::solution_update ()
  {
//       if (is_verbal_output == true)
    pcout << "    * Solution Update Step.. " << std::endl;

    vel_n_minus_1 = vel_n;
    vel_n = vel_n_plus_1;

    pre_n_minus_1 = pre_n;
    pre_n = pre_n_plus_1;

    aux_n_minus_1 = aux_n;
    aux_n = aux_n_plus_1;
  }

  template <int dim>
  void AdvectionDiffusion<dim>::vel_pre_convergence ()
  {
//     vel_res = vel_n_plus_1;
//     pre_res = pre_n_plus_1;

//     vel_res.sadd (1, -1, vel_n);
//     pre_res.sadd (1, -1, pre_n);

//     if (is_verbal_output == true)
//       pcout   << "  "
//        << vel_res.l2_norm()
//        << ", "
//        << pre_res.l2_norm()
//        << std::endl;
  }
 
  template <int dim>
  double AdvectionDiffusion<dim>::get_maximal_velocity () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
           (fe_pressure.get_degree() + 1)+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_velocity, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);
    double max_local_velocity = 0;

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      fe_values.reinit (cell);
      fe_values[velocities].get_function_values (vel_n_plus_1,
        velocity_values);

      MappingQ<dim> ff(fe_levelset.get_degree());
      double cc = VectorTools::point_value (ff,
         dof_handler_levelset,
         levelset_solution,
         cell->center());
      double dd = cc;
      cc = std::abs(cc - 0.5);
      double bb_t = 0.5 - std::pow(0.1, 1.0);
    
      if (cc < bb_t)
      for (unsigned int q=0; q<n_q_points; ++q)
 max_local_velocity = std::max (max_local_velocity, velocity_values[q].norm());
     }
    return  Utilities::MPI::max (max_local_velocity, MPI_COMM_WORLD);
  }
 
  template <int dim>
  double AdvectionDiffusion<dim>::determine_time_step ()
  {
    if (maximal_velocity < 1.0) maximal_velocity = 1.0;
   
    double critieon_1 = cfl_number*(smallest_h_size/maximal_velocity);
   
    double dum = std::pow (smallest_h_size*droplet_real_diameter, 3.0);
    double critieon_2 = std::sqrt( ((density_droplet+density_liquid)*dum) 
   / (4.*3.141592*surface_tension) );
   
     return std::min (critieon_1, critieon_2);
//    return critieon_1;
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::initial_point_concentration ()
  {
    if (is_verbal_output == true) 
      pcout <<"* Initial Concentration.." << std::endl;
    
    const Point<dim> droplet_position = ( (dim == 2) ? 
       ( Point<dim> (droplet_x_coor, droplet_y_coor)) :
         Point<dim> ( 
         droplet_x_coor, 
         droplet_y_coor,
         droplet_z_coor));
    
    MappingQ<dim> ff(fe_concentr.get_degree());
    VectorTools::interpolate ( ff,
    dof_handler_concentr,
    initial_concentration<dim> ( droplet_position, 
        droplet_radius, 
        1.0, 
        ratio_soluv),
    concentr_solution);
    
    old_concentr_solution = concentr_solution;
  }

  template <int dim>
  void AdvectionDiffusion<dim>::assemble_concentr_system ()
  {
    if (is_verbal_output == true)
      pcout << "    * Assemble system for C..." << std::endl;

    const bool use_bdf2_scheme = (timestep_number != 0);

    rhs_concentr = 0;
    matrix_concentr = 0;

    double nu_max = 0.0;

    const QGauss<dim> quadrature_formula(fe_concentr.get_degree()+1);

    FEValues<dim>     fe_concentr_values (fe_concentr, quadrature_formula,
         update_values    |
         update_gradients |
         update_hessians  |
         update_quadrature_points  |
         update_JxW_values);

    FEValues<dim>     fe_values_vel (fe_velocity, quadrature_formula,
          update_values    |
          update_gradients );

    const unsigned int   dofs_per_cell   = fe_concentr.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    Vector<double>       local_rhs (dofs_per_cell);
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   local_stiffness_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<Tensor<1,dim> > velocity_values (n_q_points);
    std::vector<Tensor<1,dim> > old_velocity_values (n_q_points);

    std::vector<double>         concentr_values (n_q_points);
    std::vector<double>         old_concentr_values(n_q_points);
    std::vector<Tensor<1,dim> > concentr_grads(n_q_points);
    std::vector<Tensor<1,dim> > old_concentr_grads(n_q_points);
    std::vector<double>         concentr_laplacians(n_q_points);
    std::vector<double>         old_concentr_laplacians(n_q_points);

    std::vector<double>         phi_T      (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);

    const std::pair<double,double>
      global_c_range = get_extrapolated_concentr_range();

    const double average_concentr = 0.5 * (global_c_range.first +
         global_c_range.second);
    const double global_entropy_variation =
      get_entropy_variation_concentr (average_concentr);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();

    typename DoFHandler<dim>::active_cell_iterator
      velocity_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++velocity_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
      local_rhs = 0;
      local_mass_matrix = 0;
      local_stiffness_matrix = 0;

      fe_concentr_values.reinit (cell);
      fe_values_vel.reinit (velocity_cell);

      fe_concentr_values.get_function_values (concentr_solution,
           concentr_values);

      fe_concentr_values.get_function_values (old_concentr_solution,
           old_concentr_values);

      fe_concentr_values.get_function_gradients (concentr_solution,
        concentr_grads);
      fe_concentr_values.get_function_gradients (old_concentr_solution,
        old_concentr_grads);

      fe_concentr_values.get_function_laplacians (concentr_solution,
        concentr_laplacians);

      fe_concentr_values.get_function_laplacians (old_concentr_solution,
        old_concentr_laplacians);

      fe_values_vel[velocities].get_function_values (vel_n_plus_1,
            velocity_values);

      fe_values_vel[velocities].get_function_values (vel_n,
            old_velocity_values);

      double nu = 0.0;
      double cc = 0.0;

      MappingQ<dim> ff(fe_levelset.get_degree());
      cc = VectorTools::point_value ( ff,
     dof_handler_levelset,
     levelset_solution,
     cell->center());
      if (cc>1.0) cc = 1.0;
      if (cc<0.0) cc = 0.0;

      double regul_diffusivity = diffusivity_liquid;
      double reg_time_step_upl = time_step_upl;

      double df_droplet = diffusivity_droplet*sqrt(ratio_soluv);
      double df_liq = diffusivity_liquid/sqrt(ratio_soluv);

      regul_diffusivity =  df_liq + (df_droplet - df_liq)*cc;
      
      double Peclet_number = velocity_ref*length_ref/regul_diffusivity;
      double DiffParameter = 1./Peclet_number;

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 Tensor<1,dim>  vel_droplet = velocity_values[q]*sqrt(ratio_soluv);
 Tensor<1,dim>  vel_liq = velocity_values[q]/sqrt(ratio_soluv);
 velocity_values[q] =  vel_liq + (vel_droplet - vel_liq)*cc;

 Tensor<1,dim>  vel_old_droplet = old_velocity_values[q]*sqrt(ratio_soluv);
 Tensor<1,dim>  vel_old_liq = old_velocity_values[q]/sqrt(ratio_soluv);

 old_velocity_values[q] = vel_old_liq +(vel_old_droplet - vel_old_liq)*cc;
      }

      reg_time_step_upl = 0.0;
      if (cc > 0.5) reg_time_step_upl = time_step_upl/sqrt(ratio_soluv);
      if (cc < 0.5) reg_time_step_upl = time_step_upl*sqrt(ratio_soluv);

      double old_reg_time_step_upl = reg_time_step_upl;
      double coef_a1 = 1.0; 
      double coef_a2 = reg_time_step_upl;

      double temp_stabilization_c_R = stabilization_c_R;
      stabilization_c_R = stabilization_c_R*3.0;
      
      nu = compute_viscosity( concentr_values,
    old_concentr_values,
    concentr_grads,
    old_concentr_grads,
    concentr_laplacians,
    old_concentr_laplacians,
    velocity_values,
    old_velocity_values,
    maximal_velocity,
    global_c_range.second - global_c_range.first,
    0.5*(global_c_range.second + global_c_range.first),
    global_entropy_variation,
    cell->diameter());

      stabilization_c_R = temp_stabilization_c_R;
      
      nu_max = std::max (nu_max, nu);
      if (use_bdf2_scheme == true)
 coef_a1 = (2*reg_time_step_upl + old_reg_time_step_upl) /
    (reg_time_step_upl + old_reg_time_step_upl);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   grad_phi_T[k] = fe_concentr_values.shape_grad (k,q);
   phi_T[k]      = fe_concentr_values.shape_value (k, q);
 }

 const double Ts
   =   (use_bdf2_scheme ?
       (concentr_values[q] *
       (reg_time_step_upl + old_reg_time_step_upl) / old_reg_time_step_upl
         -
       old_concentr_values[q] *
       (reg_time_step_upl * reg_time_step_upl) /
       (old_reg_time_step_upl * (reg_time_step_upl + old_reg_time_step_upl)))
         :
       concentr_values[q]);

      const Tensor<1,dim> ext_grad_T
   =   (use_bdf2_scheme ?
       (concentr_grads[q] *
       (1+reg_time_step_upl/old_reg_time_step_upl)
         -
       old_concentr_grads[q] *
       reg_time_step_upl / old_reg_time_step_upl)
         :
       concentr_grads[q]);

      const Tensor<1,dim> extrapolated_u
   =   (use_bdf2_scheme ?
       (velocity_values[q] * (1+reg_time_step_upl/old_reg_time_step_upl) -
       old_velocity_values[q] * reg_time_step_upl/old_reg_time_step_upl)
         :
       velocity_values[q]);

 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
     local_mass_matrix(i,j) +=  coef_a1 *
     phi_T[i] *
     phi_T[j] *
     fe_concentr_values.JxW(q);

     local_stiffness_matrix(i,j) += coef_a2 *
      DiffParameter *
      grad_phi_T[i] *
      grad_phi_T[j] *
      fe_concentr_values.JxW(q);
   }

   local_rhs(i) += (Ts * phi_T[i]
       -
     reg_time_step_upl *
     extrapolated_u * ext_grad_T * phi_T[i]
       -
     reg_time_step_upl *
     nu * ext_grad_T * grad_phi_T[i])*
     fe_concentr_values.JxW(q);
 }
      }

      cell->get_dof_indices (local_dofs_indices);

      constraint_concentr.distribute_local_to_global (local_mass_matrix,
       local_dofs_indices,
       matrix_concentr);

      constraint_concentr.distribute_local_to_global (local_stiffness_matrix,
       local_dofs_indices,
       matrix_concentr);

      constraint_concentr.distribute_local_to_global (local_rhs,
       local_dofs_indices,
       rhs_concentr);
    }

    if (is_verbal_output == true) pcout << "    * Max EV = " << nu_max << std::endl;
    
  }

  template <int dim>
  void AdvectionDiffusion<dim>::concentr_solve ()
  {
    if (is_verbal_output == true) 
      pcout << "    * Solve for C... ";
    unsigned int n_c = dof_handler_concentr.n_dofs();
    unsigned int local_dofs =
    DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distibuted_solution (map);

    matrix_concentr.compress(VectorOperation::add);
    rhs_concentr.compress(VectorOperation::add);

    SolverControl solver_control (matrix_concentr.m(),
      error_concentr*rhs_concentr.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (matrix_concentr);

    cg.solve (matrix_concentr, distibuted_solution, rhs_concentr, preconditioner);
    concentr_solution = distibuted_solution;
    constraint_concentr.distribute (concentr_solution);

    if (is_verbal_output == true)
      pcout   << solver_control.last_step() << std::endl;

  }
 
  template <int dim>
  void AdvectionDiffusion<dim>::recover_org_concentration ()
  {
    if (is_verbal_output == true) 
      pcout << "    * Recover Concentration.." << std::endl;
    
    for (unsigned int i=0; i<concentr_solution.size(); ++i)
    {
      if(levelset_solution(i)>0.5)
 org_concentration (i) = concentr_solution(i)*sqrt(ratio_soluv);
      if(levelset_solution(i)<0.5)
 org_concentration (i) = concentr_solution(i)/sqrt(ratio_soluv);

//       if (org_concentration(i) > 1.0) org_concentration (i) = 1.0;
//       if (org_concentration(i) < 0.0) org_concentration (i) = 0.0;
    }
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::set_constant_mode_concentration ()
  {
    if (is_verbal_output == true) 
      pcout << "    * Set Const. Concentration.." << std::endl;

    for (unsigned int i=0; i<concentr_solution.size(); ++i)
    {
      if(levelset_solution(i)>0.5) concentr_solution (i) = 1.0;
    }
    
  }
  
  template <int dim>
  std::pair<unsigned int, double>
  AdvectionDiffusion<dim>::DistPar(Point<dim> &coor, unsigned int number_of_particles)
  {
    unsigned int q1 = 9000;
    double q2 = 9000;

    for (unsigned int n = 0 ; n < number_of_particles ; ++n)
    {
      double tt = particle_position[n].distance(coor) - 0.5*par_diameter;

      if (tt < q2)
      {
 q1 = n;
 q2 = tt;
      }

    }
    return std::make_pair(q1, q2);
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::particle_solution ()
  {
    if (is_verbal_output == true)
      pcout << "  # Particle Position.. ";
    
    matrix_levelset = 0;
    rhs_levelset = 0;

    const QGauss<dim> quadrature_formula(fe_levelset.get_degree()+1 + 1);

    FEValues<dim> fe_values_levelset (fe_levelset, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_levelset.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_levelset.begin_active(),
    endc = dof_handler_levelset.end();

    for (; cell!=endc; ++cell)
        if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_levelset.reinit (cell);
            cell->get_dof_indices (local_dofs_indices);

            local_matrix = 0;
            local_rhs = 0;

     Point<dim> c = cell->center();
     std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);
 
     double q2 = distant_of_par.second;
     double tt_d = 1./(1. + std::exp( (q2)/(0.5*eps_v_levelset)));
     
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
      local_matrix(i,j) +=  fe_values_levelset.shape_value(i, q)*
         fe_values_levelset.shape_value(j, q)*
         fe_values_levelset.JxW(q);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    local_rhs(i) += fe_values_levelset.shape_value(i, q)*
     tt_d*
     fe_values_levelset.JxW(q);

            }

            constraint_levelset.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_levelset);

            constraint_levelset.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_levelset);
        }


    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_xx (map);

    matrix_levelset.compress(VectorOperation::add);
    rhs_levelset.compress(VectorOperation::add);

    SolverControl solver_control (matrix_levelset.m(), error_vel*rhs_levelset.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_levelset);
    cg.solve (matrix_levelset, distibuted_xx, rhs_levelset, preconditioner);
    particle_at_levelset = distibuted_xx;

    pcout   << solver_control.last_step()
            << std::endl;

    constraint_levelset.distribute (particle_at_levelset);  
  }

  template <int dim>
  void AdvectionDiffusion<dim>::pars_move (std::ofstream &out_ppp)
  {
    if (num_of_par == 0) return;
    
    if (is_verbal_output == true)
      pcout << "  # Particle Movement.. " << std::endl;
    
    QGauss<dim>  quadrature_formula (fe_levelset.get_degree()+1);
    const unsigned int n_q_points = quadrature_formula.size();;
    const unsigned int dofs_per_cell = fe_levelset.dofs_per_cell;
    
    FEValues<dim> fe_velocity_values (fe_velocity, quadrature_formula,
     UpdateFlags(update_values    |
     update_gradients |
     update_quadrature_points  |
     update_JxW_values));

    FEValues<dim> fe_values_levelset (fe_levelset, quadrature_formula,
     update_values    |
     update_quadrature_points  |
     update_JxW_values |
     update_gradients);
     
    typename DoFHandler<dim>::active_cell_iterator  cell, endc;
      cell = dof_handler_levelset.begin_active();
      endc = dof_handler_levelset.end();

    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();
      
    std::vector<Tensor<1,dim> > vel_solu (n_q_points);
    std::vector<Tensor<2,dim> > vel_solGrads (n_q_points);
    
    std::vector<Point<dim> >  totVel;
    std::vector<double>  totvis;

    const FEValuesExtractors::Vector velocities (0);
    
    for (unsigned int n=0; n<num_of_par; ++n)
    {
      Point<dim> aa;
      double bb = 0.0;
      totVel.push_back (aa);
      totvis.push_back (bb);
    }

    for (; cell!=endc; ++cell, ++vel_cell)
    {
      fe_values_levelset.reinit (cell);
      fe_velocity_values.reinit (vel_cell);
      fe_velocity_values[velocities].get_function_values (vel_n_plus_1, vel_solu);
      fe_velocity_values[velocities].get_function_gradients (vel_n_plus_1, vel_solGrads);
     
      Point<dim> c = cell->center();
      std::pair<unsigned int,double> distant_of_par = DistPar (c, num_of_par);

      if (distant_of_par.second <0)
      for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
 unsigned int nn_par = distant_of_par.first;

 totvis[nn_par] += fe_values_levelset.shape_value (i,q) *
    fe_values_levelset.JxW(q);
 for (unsigned d=0; d<dim; ++d)
   totVel[nn_par][d] += fe_values_levelset.shape_value (i,q) *
    fe_values_levelset.JxW(q)*
    vel_solu[q][d];
      }
    }

    out_ppp << timestep_number*time_step_upl;
    for (unsigned n=0; n<num_of_par; ++n)
    {
      for (unsigned int d=0; d<dim; ++d)
 particle_position[n][d] += (totVel[n][d] / totvis[n]) * time_step_upl;
  
//       if (is_verbal_output == true)
//  pcout << "  "
//   << n << " "
//   << totvis[n] << " "
//   << totVel[n] << " "
//   << particle_position[n] << " "
//   << std::endl;
   
      out_ppp << " " << n << " " << particle_position[n];
    }
    out_ppp << std::endl;
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::initialize_for_mass_transfer_parameter ()
  {
    intercell_no = 0;
    intercell_no_velNorm = 0;
    sum_of_mass_normal_flux1 = 0;
    sum_mass_in_liquid_at_interface = 0;
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::droplet_parameter(std::ofstream &out_d5)
  {    
    if (is_verbal_output == true)
 pcout <<"*   Compute Droplet Parameter...";
    
    avr_droplet_size = 0.0; 
    deformability = 0.0;
    avr_droplet_vel = ( (dim == 2) ? ( Point<dim> (0.0, 0.0)) :
     Point<dim> (0.0, 0.0, 0.0));
    avr_droplet_pos = ( (dim == 2) ? ( Point<dim> (0.0, 0.0)) :
     Point<dim> (0.0, 0.0, 0.0));
    Point<dim> avr_domain_vel = ( (dim == 2) ? (Point<dim> (0.0, 0.0)) :
      Point<dim> (0.0, 0.0, 0.0));
    
    const QGauss<dim> quadrature_formula (fe_levelset.get_degree()+1);
    FEValues<dim>  fe_values_levelset (fe_levelset, quadrature_formula,
                                               update_values    |
                                               update_gradients |
                                               update_hessians  |
                                               update_quadrature_points  |
                                               update_JxW_values);

    FEValues<dim> fe_values_velocity (fe_velocity,
         quadrature_formula,
         update_values |
         update_quadrature_points);

    const unsigned int   dofs_per_cell   = fe_levelset.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<double>         phi_T (dofs_per_cell);
    std::vector<double> level_set_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler_levelset.begin_active(),
   endc = dof_handler_levelset.end();

    typename DoFHandler<dim>::active_cell_iterator
   vel_cell = dof_handler_velocity.begin_active();

    std::vector<Point<dim> > sp (dof_handler_levelset.n_dofs());
    MappingQ<dim> ff (fe_levelset.get_degree());
    DoFTools::map_dofs_to_support_points (ff, dof_handler_levelset, sp);

    unsigned int count = 0;
    unsigned int num_ele = 0;
    std::vector<Point<dim> > leng_de;
    std::vector<bool> is_leng_de;
    {
      Point<dim> aa;
      bool bb = false;
      for (unsigned int i=0; i<triangulation.n_active_cells(); ++i)
      {
 leng_de.push_back (aa);
 is_leng_de.push_back (bb);
      }
    }

    for (; cell!=endc; ++cell, ++vel_cell, ++num_ele)
    {
      fe_values_levelset.reinit (cell);
      fe_values_velocity.reinit (vel_cell);

      fe_values_velocity[velocities].get_function_values (vel_n_plus_1, vel_values);

      MappingQ<dim> ff(fe_levelset.get_degree());
      double qq = VectorTools::point_value (ff,
         dof_handler_levelset,
         levelset_solution,
         cell->center());
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {
 for (unsigned int k=0; k<dofs_per_cell; ++k)
   phi_T[k] = fe_values_levelset.shape_value (k, q);
 
 for (unsigned int i=0; i<dofs_per_cell; ++i)
 {
//    if (qq > 0.4 && qq < 0.6)
//    avr_domain_vel += phi_T[i]*
//        vel_values[q]*
//        fe_values_levelset.JxW(q);
   if (qq > 0.5) 
   {
     avr_droplet_size += phi_T[i]*
    fe_values_levelset.JxW(q);
         
     avr_droplet_vel +=  phi_T[i]*
    vel_values[q]*
    fe_values_levelset.JxW(q);

     avr_droplet_pos +=  phi_T[i]*
    cell->center()*
    fe_values_levelset.JxW(q);
   }
 }
      }
    }

    avr_droplet_pos /= avr_droplet_size;
    avr_droplet_vel /= avr_droplet_size;

    double min_d = 9999;
    double max_d = 1e-10;

    for (unsigned int i=0; i<dof_handler_levelset.n_dofs(); ++i)
    {
      if (levelset_solution(i) < 0.5+eps_v_levelset &&
   levelset_solution(i) > 0.5-eps_v_levelset)
      {
 double dist = sp[i].distance (avr_droplet_pos);
 if (min_d > dist) min_d = dist;
 if (max_d < dist) max_d = dist;
      }
    }

    deformability = std::abs( (max_d - min_d)/(max_d + min_d) );
    
    out_d5  << timestep_number << " "
  << total_time*time_ref << " "
  << avr_droplet_size << " "
  << avr_droplet_vel << " "
  << max_d << " "
  << min_d << " "
  << deformability << " "
  << std::endl;
    pcout << std::endl;
  }


  template <int dim>
  void AdvectionDiffusion<dim>::find_midPoint_on_interCell (std::vector<bool> &touch_interface_on_cell,
            std::vector<Point<dim> > &line_seg_points_at_interface)
  {
    if (is_verbal_output == true)
      pcout << "*   Find Mid-Point On Interface Cell... " << std::endl;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();

    unsigned int cell_no = 0;

    for (; cell!=endc; ++cell, ++cell_no)
    if (touch_interface_on_cell[cell_no] == true)
    {
     std::vector<Point<dim> > ddd;
     Point<dim> dum_a;
     for (unsigned int kk=0; kk<2; ++kk)
       ddd.push_back(dum_a);

     unsigned int no_b1_b2 = 0;

     for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
     {
  unsigned int a1 = cell->face(f)->vertex_dof_index(0, 0);
  unsigned int a2 = cell->face(f)->vertex_dof_index(1, 0);

  double b1 = levelset_solution (a1) - 0.5;
  double b2 = levelset_solution (a2) - 0.5;

  Point<dim> c1, c2;

  Point<dim> dum;

  if (b1*b2 < 0.0)  // interface segment
  {
      c1 = cell->face(f)->vertex(0);
      c2 = cell->face(f)->vertex(1);

      double e1 = std::abs(b1);
      double e2 = std::abs(b2);

      double e3 = e1+e2;

      switch (f) {
     case 0 :
     {
         dum[0] = c1[0];
         dum[1] = c1[1] + (e1/e3)*abs(c2[1]-c1[1]);

         if (c1[1] > c2[1])
    dum[1] = c2[1] + c2[1]*(e2/e3)*abs(c2[1]-c1[1]);

         break;
     }
     case 1 :
     {
         dum[0] = c1[0];
         dum[1] = c1[1] + (e1/e3)*abs(c2[1]-c1[1]);

         if (c1[1] > c2[1])
    dum[1] = c2[1] + c2[1]*(e2/e3)*abs(c2[1]-c1[1]);

         break;
     }
     case 2 :
     {
         dum[1] = c1[1];
         dum[0] = c1[0] + (e1/e3)*abs(c2[0]-c1[0]);

         if (c1[0] > c2[0])
    dum[0] = c2[0] + c2[0]*(e2/e3)*abs(c2[0]-c1[0]);

         break;
     }
     case 3 :
     {
         dum[1] = c1[1];
         dum[0] = c1[0] + (e1/e3)*abs(c2[0]-c1[0]);

         if (c1[0] > c2[0])
    dum[0] = c2[0] + c2[0]*(e2/e3)*abs(c2[0]-c1[0]);

         break;
     }
      } //switch

      if (no_b1_b2 == 2)
        if (is_verbal_output == true)
   pcout << "Error.. Error" << std::endl;

      ddd[no_b1_b2] = dum;

      ++no_b1_b2;
  } //b1*b2<0
     }//for-face


     for (unsigned int kk=0; kk<2; ++kk)
     {
  bool multi_pli = false;
  for (unsigned int jj = 0; jj<intercell_no; ++jj)
    if (std::abs(ddd[kk][0] - line_seg_points_at_interface [jj][0]) < 1e-8
      && std::abs(ddd[kk][1] - line_seg_points_at_interface [jj][1]) < 1e-8)
    {
        multi_pli = true;
    }

    if (multi_pli == false)
    {
        for (unsigned int d=0; d<dim; ++d)
     line_seg_points_at_interface [intercell_no][d] = ddd[kk][d];

        ++intercell_no;
    }
     }

 }

 //for-validation
//      for (unsigned int kk=0; kk<intercell_no; ++kk)
//      {
//         MappingQ<dim> ff(fe_levelset.get_degree());
//         double rer = VectorTools::point_value (ff,
//              dof_handler_levelset,
//              levelset_solution,
//              line_seg_points_at_interface [kk]);

//         pcout << cell_no << " " << intercell_no << " "
//         << line_seg_points_at_interface [kk] << " " << rer << std::endl;
//      }
  }

  template <int dim>
  void AdvectionDiffusion<dim>::compute_normal_grad_mass(std::vector<bool> &touch_interface_on_cell)
  {
    if (is_verbal_output == true)
      pcout << "*   Compute Normal Grad...";

    rhs_concentr = 0;
    TrilinosWrappers::MPI::Vector  rhs_concentr2 (rhs_concentr);
    matrix_concentr = 0;

    const QGauss<dim> quadrature_formula (fe_concentr.get_degree()+1);

    FEValues<dim>     fe_concentr_values (fe_concentr, quadrature_formula,
                                               update_values    |
                                               update_gradients |
                                               update_hessians  |
                                               update_quadrature_points  |
                                               update_JxW_values);

    FEValues<dim>     fe_values_level (fe_levelset, quadrature_formula,
                                               update_values    |
                                               update_gradients );

    FEValues<dim>     fe_values_vel (fe_velocity, quadrature_formula,
                                               update_values    |
                                               update_gradients );

    const unsigned int   dofs_per_cell   = fe_concentr.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    Vector<double>       local_rhs (dofs_per_cell);
    Vector<double>       local_rhs2 (dofs_per_cell);

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    std::vector<Tensor<1,dim> > concentr_grads (n_q_points);

    std::vector<double>         phi_T      (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);

    std::vector<double> level_nrm_x  (n_q_points);
    std::vector<double> level_nrm_y  (n_q_points);
    std::vector<double> level_nrm_z  (n_q_points);

    std::vector<Tensor<1,dim> > velocity_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler_concentr.begin_active(),
   endc = dof_handler_concentr.end();

    typename DoFHandler<dim>::active_cell_iterator
   levelset_cell = dof_handler_levelset.begin_active();

    typename DoFHandler<dim>::active_cell_iterator
   velocity_cell = dof_handler_velocity.begin_active();

    unsigned int count_ele = 0;
    const FEValuesExtractors::Vector velocities (0);
     
    unsigned int n_c = dof_handler_concentr.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::Vector concentr_liq (map);
    TrilinosWrappers::MPI::Vector distibuted_solution (map);

    for (unsigned int k=0; k<concentr_liq.size(); ++k)
 concentr_liq(k) = concentr_solution(k)/sqrt(ratio_soluv);
    
    for (; cell!=endc; ++cell, ++levelset_cell, ++velocity_cell, ++count_ele)
//     if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
 cell->get_dof_indices (local_dofs_indices);

 local_rhs = 0;
 local_rhs2 = 0;
 local_matrix = 0;

 bool at_interface_on_cell = false;
 unsigned int which_phase_ini_node = 0;

        for (unsigned int k=0; k<dofs_per_cell; ++k)
 {
   double level_value_at_node = levelset_solution (local_dofs_indices[k]) - 0.5;

   if (k == 0)
   {
     if (level_value_at_node < 0) which_phase_ini_node = 1;
     if (level_value_at_node > 0) which_phase_ini_node = 2;
   }

   if (k > 0)
   {
     if (which_phase_ini_node == 1 && level_value_at_node > 0)
       at_interface_on_cell = true;

     if (which_phase_ini_node == 2 && level_value_at_node < 0)
       at_interface_on_cell = true;
   }
 }

 fe_concentr_values.reinit (cell);
 fe_values_level.reinit (levelset_cell);
 fe_values_vel.reinit (velocity_cell);

 fe_concentr_values.get_function_gradients (concentr_solution,
          concentr_grads);

 fe_values_vel[velocities].get_function_values ( vel_n_plus_1,
            velocity_values);

 fe_values_level.get_function_values (level_set_normal_x, level_nrm_x);
 fe_values_level.get_function_values (level_set_normal_y, level_nrm_y);
 if (dim == 3)
     fe_values_level.get_function_values (level_set_normal_z, level_nrm_z);

 if (at_interface_on_cell == true) touch_interface_on_cell[count_ele] = true;

 for (unsigned int q=0; q<n_q_points; ++q)
 {
   Point<dim> levelset_normal;

   levelset_normal[0] = -level_nrm_x[q];
   levelset_normal[1] = -level_nrm_y[q];
   if (dim == 3)
     levelset_normal[2] = -level_nrm_z[q];

   for (unsigned int k=0; k<dofs_per_cell; ++k)
   {
     grad_phi_T[k] = fe_concentr_values.shape_grad (k,q);
     phi_T[k]      = fe_concentr_values.shape_value (k, q);
   }

   for (unsigned int i=0; i<dofs_per_cell; ++i)
   {
     for (unsigned int j=0; j<dofs_per_cell; ++j)
       local_matrix(i,j) += phi_T[i]*
        phi_T[j]*
        fe_concentr_values.JxW(q);
     double yz0 = 0.0;
     double yz1 = 0.0;
     double yy = 0.0;
      
     for (unsigned int d=0; d<dim; ++d)
       yy += velocity_values[q][d]*levelset_normal[d];

     for (unsigned int d=0; d<dim; ++d)
       yz0 += std::abs((-concentr_grads[q][d])*levelset_normal[d]) ;

     if (yy>0 && at_interface_on_cell == true)
     for (unsigned int d=0; d<dim; ++d)
       yz1 += std::abs((-concentr_grads[q][d])*levelset_normal[d]) ;

     local_rhs(i) += fe_concentr_values.shape_value (i, q)*
    yz0*
    fe_concentr_values.JxW (q);

     local_rhs2(i) += fe_concentr_values.shape_value (i, q)*
    yz1*
    fe_concentr_values.JxW (q);

   }
 }

 constraint_concentr.distribute_local_to_global ( local_matrix,
           local_dofs_indices,
           matrix_concentr);

 constraint_concentr.distribute_local_to_global ( local_rhs,
           local_dofs_indices,
           rhs_concentr);

 constraint_concentr.distribute_local_to_global ( local_rhs2,
           local_dofs_indices,
           rhs_concentr2);
    }

    matrix_concentr.compress (VectorOperation::add);
    rhs_concentr.compress (VectorOperation::add);
    rhs_concentr2.compress (VectorOperation::add);

    SolverControl solver_control (matrix_concentr.m(),
                                      1e-11*rhs_concentr.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (matrix_concentr);

    {
      cg.solve (matrix_concentr, distibuted_solution,
  rhs_concentr, preconditioner);
      normal_vec_out_flux_concentr = distibuted_solution;
      constraint_concentr.distribute (normal_vec_out_flux_concentr);
    }

    {
      distibuted_solution = 0;
      cg.solve (matrix_concentr, distibuted_solution,
  rhs_concentr2, preconditioner);

      normal_vec_out_flux_concentr2 = distibuted_solution;
      constraint_concentr.distribute (normal_vec_out_flux_concentr2);
    }
    
    if (is_verbal_output == true)
      pcout   << solver_control.last_step()
       << std::endl;
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::mass_transfer_parameter
     (std::vector<Point<dim> > &line_seg_points_at_interface,
      std::ofstream &out_d3)
  { 
    if (is_verbal_output == true)
 pcout <<"*   Compute Mass Transfer Parameter..." << std::endl;

    double length_at_int = sqrt(2);

    for (unsigned int i=0; i<intercell_no; ++i)
    {
 Point<dim> trnC_liq_pos, trnC_gas_pos, norm_Vec;
 Vector<double> vel_at_intPoint(dim);

 //find the normal vector
 MappingQ<dim> ff(fe_levelset.get_degree());
 MappingQ<dim> ddf(fe_pressure.get_degree()+1);

 norm_Vec[0] = VectorTools::point_value (ff,
      dof_handler_levelset,
       level_set_normal_x,
       line_seg_points_at_interface [i]);

 norm_Vec[1] = VectorTools::point_value (ff,
      dof_handler_levelset,
       level_set_normal_y,
       line_seg_points_at_interface [i]);

  VectorTools::point_value (ddf,
     dof_handler_velocity,
      vel_n_plus_1,
      line_seg_points_at_interface [i],
    vel_at_intPoint);


 double sing_nor_vel = 0;
 for (unsigned int d=0; d<dim; ++d)
   sing_nor_vel += vel_at_intPoint[d]*norm_Vec[d];

 if (sing_nor_vel > 0)
 {
     ++intercell_no_velNorm;

     for (unsigned int d=0; d<dim; ++d)
  trnC_liq_pos[d] = line_seg_points_at_interface [i][d]+
      length_at_int*smallest_h_size*(-norm_Vec[d]);

     for (unsigned int d=0; d<dim; ++d)
  trnC_gas_pos[d] = line_seg_points_at_interface [i][d] +
      length_at_int*smallest_h_size*(norm_Vec[d]);

     MappingQ<dim> gg(fe_concentr.get_degree());
     double value_trnC_in_liq =  VectorTools::point_value (gg,
          dof_handler_concentr,
          concentr_solution,
          trnC_liq_pos);

     double value_trnC_in_gas = VectorTools::point_value (gg,
          dof_handler_concentr,
          concentr_solution,
          trnC_gas_pos);

     double xxx = VectorTools::point_value (ff,
          dof_handler_concentr,
          concentr_solution,
          line_seg_points_at_interface [i]);

     sum_mass_in_liquid_at_interface += xxx/ratio_soluv;

     MappingQ<dim> dde (fe_concentr.get_degree());
     double rer = VectorTools::point_value ( dde,
          dof_handler_concentr,
          normal_vec_out_flux_concentr,
          line_seg_points_at_interface [i]);

     sum_of_mass_normal_flux1 += rer/ratio_soluv;

 }
    }  
    
    out_d3 << timestep_number << " "
  << total_time*time_ref << " "
  << intercell_no << " "
  << intercell_no_velNorm << " "
  << sum_of_mass_normal_flux1 << " "
  << sum_mass_in_liquid_at_interface << " "
  << std::endl;
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::compute_effective_viscosity ( unsigned int iii,
             std::ofstream &out_q1)
  {
    pcout <<"*   Compute Effective Viscosity..." << std::endl;

    std::ostringstream filename_ev;
    filename_ev << "data_files/ev/" << "points_ev_" << Utilities::int_to_string(iii, 4) << ".dat";
    std::ofstream out_ev (filename_ev.str().c_str());

    const unsigned int num_check_ev = 7;
    std::vector<double> global_effective_viscosity (num_check_ev); 
    std::vector<unsigned int> counter_ev (num_check_ev);
    std::vector<std::vector<Point<dim> > > find_points_where_for_ev (num_check_ev,
            std::vector<Point<dim> > (1000000));
    
    const QIterated<dim> quadrature_formula (QTrapez<1>(), fe_pressure.get_degree()+2);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_velocity_values ( fe_velocity,quadrature_formula,
     UpdateFlags(
     update_values    |
     update_gradients |
     update_q_points  |
     update_JxW_values));
    
    FEValues<dim> fe_levelset_values ( fe_levelset,quadrature_formula,
     UpdateFlags(
     update_values));

    const unsigned int   dofs_per_cell = fe_velocity.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    std::vector<Tensor<1,dim> > velocity_solu (n_q_points);
    std::vector<Tensor<2,dim> > velocity_solGrads (n_q_points);
    std::vector<double>  levelset_solu (n_q_points);
    
    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end(),
      level_cell = dof_handler_levelset.begin_active();
    
    for (; cell!=endc; ++cell, ++level_cell)
    {
      fe_velocity_values.reinit (cell);
      fe_levelset_values.reinit (level_cell);
      fe_levelset_values.get_function_values (levelset_solution, levelset_solu);
      fe_velocity_values[velocities].get_function_values (vel_n_plus_1, velocity_solu);
      fe_velocity_values[velocities].get_function_gradients (vel_n_plus_1, velocity_solGrads);
      
      cell->get_dof_indices (local_dof_indices);
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {
 double dimless_viscosity = 1.0;
 
 for (unsigned int i=0; i<num_check_ev; ++i)
 {   
   double cc = 0.5 - levelset_solu[q];

   if (cc >= 0.0)
   {
     
     double bb_t = 1e-2;
     if (i > 0) bb_t = 0.5 - std::pow(0.1, double(i));
     
     std::vector<Point<dim> > quad_points = 
          fe_velocity_values.get_quadrature_points();   
     if (num_of_par > 0)
     {
       std::pair<unsigned int,double> distant_of_par = DistPar (quad_points[q], num_of_par);
   
       if (distant_of_par.second < 0.0)
  dimless_viscosity = factor_visocisty;
     }
   
     if (std::abs(cc) < bb_t)
     {
       counter_ev[i] = counter_ev[i] + 1;
       
       for (unsigned int d=0; d<dim; ++d)
  find_points_where_for_ev [i][counter_ev[i]][d] = quad_points[q][d] - avr_droplet_pos[d];
      
       double tt = velocity_solGrads[q][0][1]*velocity_solGrads[q][0][1]
     +
     velocity_solGrads[q][1][0]*velocity_solGrads[q][1][0];
       tt = std::sqrt(tt);
       
       global_effective_viscosity[i] +=  dimless_viscosity*
      tt*
      fe_velocity_values.JxW(q);
     }
   }
 }
      }
    }
  
    out_q1 << timestep_number;
    for (unsigned int i=0; i<7; ++i)
      out_q1 << " " << global_effective_viscosity [i];
    out_q1 << std::endl;  
    
    unsigned int max_counter_cv = *std::max_element(counter_ev.begin(), counter_ev.end());
        
    for (unsigned int n=0; n<counter_ev.size(); ++n)
      out_ev << counter_ev[n] << " ";
    out_ev << std::endl;
 
    for (unsigned int i=0; i<max_counter_cv; ++i)
    {
      for (unsigned int n=0; n<counter_ev.size(); ++n)
      {
 out_ev << find_points_where_for_ev [n][i] << " ";
      }
      out_ev << std::endl;
    }
  }
        
  template <int dim>
  void AdvectionDiffusion<dim>::plotting_solution (unsigned int np)
  {
    if (is_verbal_output == true)
    pcout << "  # Output Flow.. "  << np << std::endl;
    
    if (num_of_par >0) particle_solution ();
    
    const FESystem<dim> joint_fe ( 
        fe_velocity, 1,
        fe_pressure, 1,
        fe_levelset, 1,
        fe_levelset, 1,
        fe_concentr, 1);

    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Vector<double> joint_solution (joint_dof_handler.n_dofs());

    {
      std::vector<unsigned int> local_joint_dof_indices (joint_fe.dofs_per_cell);
      std::vector<unsigned int> local_vel_dof_indices (fe_velocity.dofs_per_cell);
      std::vector<unsigned int> local_pre_dof_indices (fe_pressure.dofs_per_cell);
      std::vector<unsigned int> local_levelset_dof_indices (fe_levelset.dofs_per_cell);
      std::vector<unsigned int> local_concentr_dof_indices (fe_concentr.dofs_per_cell);
      
      typename DoFHandler<dim>::active_cell_iterator
 joint_cell = joint_dof_handler.begin_active(),
 joint_endc   = joint_dof_handler.end(),
 vel_cell     = dof_handler_velocity.begin_active(),
 pre_cell     = dof_handler_pressure.begin_active(),
 level_cell   = dof_handler_levelset.begin_active(),
 concentr_cell  = dof_handler_concentr.begin_active();

 for (; joint_cell!=joint_endc; ++joint_cell, ++vel_cell, ++pre_cell, ++level_cell, ++concentr_cell)
 {
   joint_cell -> get_dof_indices (local_joint_dof_indices);
   vel_cell-> get_dof_indices (local_vel_dof_indices);
   pre_cell-> get_dof_indices (local_pre_dof_indices);
   level_cell-> get_dof_indices (local_levelset_dof_indices);
   concentr_cell-> get_dof_indices (local_concentr_dof_indices);

   for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
   if (joint_fe.system_to_base_index(i).first.first == 0)
   {
     joint_solution(local_joint_dof_indices[i])
     = vel_n_plus_1 (local_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
   }
   else if (joint_fe.system_to_base_index(i).first.first == 1)
   {
     joint_solution(local_joint_dof_indices[i])
     = pre_n_plus_1 (local_pre_dof_indices[joint_fe.system_to_base_index(i).second]);
   }
   else if (joint_fe.system_to_base_index(i).first.first == 2)
   {
     joint_solution(local_joint_dof_indices[i])
     = levelset_solution (local_levelset_dof_indices[joint_fe.system_to_base_index(i).second]);
   }
   else if (joint_fe.system_to_base_index(i).first.first == 3)
   {
     joint_solution(local_joint_dof_indices[i])
     = particle_at_levelset (local_levelset_dof_indices[joint_fe.system_to_base_index(i).second]);
   }
   else if (joint_fe.system_to_base_index(i).first.first == 4)
   {
     joint_solution(local_joint_dof_indices[i])
     = org_concentration (local_concentr_dof_indices[joint_fe.system_to_base_index(i).second]);
   }
 }
    }

    std::vector<std::string> joint_solution_names (dim, "U");
    joint_solution_names.push_back ("P");
    joint_solution_names.push_back ("L");
    joint_solution_names.push_back ("PAR");
    joint_solution_names.push_back ("C");    
    
    DataOut<dim> data_out;

    data_out.attach_dof_handler (joint_dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim + 4, DataComponentInterpretation::component_is_scalar);
     
    for (unsigned int i=0; i<dim; ++i)
      data_component_interpretation[i]
      = DataComponentInterpretation::component_is_part_of_vector;

    data_out.add_data_vector (joint_solution, joint_solution_names,
         DataOut<dim>::type_dof_data,
         data_component_interpretation);
    data_out.build_patches (3);

    std::ostringstream filename;
    filename << "vtu_files/" << "solution-" << Utilities::int_to_string(np, 4) << ".vtu";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);
    
  }

  template <int dim>
  void AdvectionDiffusion<dim>::solve_levelSet_function ()
  {
    pcout << "  1. Level Set Equation..."<< std::endl;
    
    unsigned int n_i = dof_handler_levelset.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::Vector tmp_tmp_level_set (map), tmp_level_set(map);
    
    level_set_2nd_adv_step ();

    bool reinit_imle = false;
    if (timestep_number%1 == 0) reinit_imle = true;
    if (reinit_imle == true)
    {
      tmp_tmp_level_set = levelset_solution;

      for (unsigned int d=0; d<dim; ++d)
  level_set_compute_normal (d);

      unsigned int ddd = 0;
      double min_value = 0.0;
      double max_value = 0.0;
      
      do {
  tmp_level_set = levelset_solution;
  level_set_reinitial_step ();
  tmp_level_set.sadd (-1, levelset_solution);
  if (ddd%50 == 0)
  {
      min_value = levelset_solution(0);
      max_value = levelset_solution(0);

      for (unsigned int i=0; i<levelset_solution.size(); ++i)
      {
   min_value = std::min<double> (min_value,
           levelset_solution(i));
   max_value = std::max<double> (max_value,
           levelset_solution(i));
      }

//       if (is_verbal_output == true)
//         pcout << "RR "<< ddd << " th "
//      << min_value << " " << max_value << " "
//      << tmp_level_set.linfty_norm () << std::endl;
 }
   ++ddd;
      } while ( tmp_level_set.linfty_norm () > error_reinit*smallest_h_size);
      if (is_verbal_output == true)
      pcout << "    * "<< ddd << " th "
     << min_value << " " << max_value << " "
     << tmp_level_set.linfty_norm () << std::endl;

      old_levelset_solution = tmp_tmp_level_set;
    }   
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::solve_fluid_equation ()
  {
    pcout << "  2. NS Equation..."<< std::endl;

    vel_star = 0;
    vel_star.equ (2.0, vel_n, -1.0, vel_n_minus_1);

//     variable_distribution ();
    
    diffusion_step ();

    maximal_velocity = get_maximal_velocity();
    
    projection_step ();
    
    pressure_correction_step_rot ();
    
//     vel_pre_convergence ();
    
    solution_update ();    
  }
  
  template <int dim>
  void AdvectionDiffusion<dim>::solve_mass_transfer_equation ()
  {
    pcout << "  3. Mass Transfer Equation..."<< std::endl;
    
    unsigned int n_i = dof_handler_concentr.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
         Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);    
    TrilinosWrappers::Vector tmp_concentr_solution (map);
    tmp_concentr_solution = concentr_solution;
    
    assemble_concentr_system ();
    concentr_solve ();    
    
    old_concentr_solution = tmp_concentr_solution;
    
    recover_org_concentration ();
    
    if (is_set_const_mode == true) set_constant_mode_concentration ();
  }

  template <int dim>
  void AdvectionDiffusion<dim>::run ()
  {
    std::ostringstream filename_d3;
    filename_d3 << "data_files/" << "mass_transfer.dat";
    std::ofstream out_d3 (filename_d3.str().c_str());

    std::ostringstream filename_d5;
    filename_d5 << "data_files/" << "droplet.dat";
    std::ofstream out_d5 (filename_d5.str().c_str());

    std::ostringstream filename_p;
    filename_p << "data_files/" << "particle.dat";
    std::ofstream out_ppp (filename_p.str().c_str());

    std::ostringstream filename_rr;
    filename_rr << "data_files/" << "output_time.dat";
    std::ofstream out_time (filename_rr.str().c_str());
    
    std::ostringstream filename_ev;
    filename_ev << "data_files/" << "eff_viscosity.dat";
    std::ofstream out_q1 (filename_ev.str().c_str());
    
    readat ();
    initial_particle_distribution ();
    create_init_coarse_mesh ();
    create_init_adaptive_mesh ();
    
    setup_dofs (1);
    prepare_for_levelset ();
    initial_Levelset ();
    initial_point_concentration ();
    
    int index_plotting = -1;
    int index_data = -1;
    double given_time_step = time_step_upl;
    double allow_output_time = time_ref*given_time_step*output_fac;
    pcout << std::endl;
        
    plotting_solution (0);
    
    do
    {
      bool allow_print = false;
      old_time_step_upl = time_step_upl;
//        time_step_upl = determine_time_step ();
      time_step_upl = given_time_step;
      old_time_step_upl = time_step_upl;
      total_time += time_step_upl;
      
      pcout << "  "
  <<"# No. = " << timestep_number << ", "
  << total_time << ", "
  << end_time << " | "
  << time_ref*total_time << " | "
  << allow_output_time
  << std::endl;
  
      pcout << "  " 
  << "# Maximal Velocity = " 
  << maximal_velocity << " | " 
  << time_step_upl << " | "
  << old_time_step_upl
  << std::endl; 
  
      solve_levelSet_function ();
            
      for (unsigned int d=0;d<dim;++d)
 sufTen_compute_gradient (d);
      surTen_compute_curvature ();
      
      {
 unsigned int res_i = no_buffered_interval;
 double tmp_t = time_step_upl;
 double tmp_old_t = old_time_step_upl;
      
 time_step_upl = tmp_t*(1./double(res_i));
 old_time_step_upl = tmp_old_t*(1./double(res_i));
 for (unsigned int j=0; j<res_i; ++j)
 {
   pcout << "**** " << j << ", " << time_step_upl 
  << ", " << old_time_step_upl << std::endl;
   solve_fluid_equation ();
//    solve_mass_transfer_equation ();
 }
      
 time_step_upl = tmp_t;
 old_time_step_upl = tmp_old_t;
      }
      
//       initialize_for_mass_transfer_parameter ();  
//       std::vector<bool> touch_interface_on_cell (triangulation.n_active_cells(), false);
//       compute_normal_grad_mass (touch_interface_on_cell);

//       Point<dim> dum;
//       std::vector<Point<dim> > line_seg_points_at_interface (10000, dum);
//       find_midPoint_on_interCell ( touch_interface_on_cell,
//       line_seg_points_at_interface);

      if (Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0)
      {
//  droplet_parameter (out_d5);

      }
  
      if (Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0)
      {
 
 if (timestep_number%50 == 0) 
 {
   ++index_data;
//    compute_effective_viscosity (index_data, out_q1);
 }
 
//  mass_transfer_parameter (line_seg_points_at_interface, out_d3);
      
 if ( (allow_output_time < time_ref*total_time) 
     ||
     timestep_number == 0)
 {
   allow_print = true;
   ++index_plotting;
   allow_output_time = time_ref*given_time_step*output_fac*(index_plotting+1);
 
   plotting_solution (index_plotting);
 }
      }
      

 
      if (timestep_number%refine_fac == 0) refine_mesh ();
      
      pars_move (out_ppp);
      //Reset needed
      particle_at_levelset = 0;
      interface_on_cell = 0;
      intercell_no_velNorm = 0;
      
      ++timestep_number; 
      pcout << std::endl;
    } while (time_ref*total_time < end_time);
  }


int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;

        deallog.depth_console (0);

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                            numbers::invalid_unsigned_int);

        ParameterHandler  prm;
        ParameterReader   param(prm);
        param.read_parameters("input.prm");

        prm.enter_subsection ("Problem Definition");
        unsigned int dimn = prm.get_integer ("Dimension");
        prm.leave_subsection ();

        if (dimn == 2)
        {
            AdvectionDiffusion<2>  ysh (prm);
            ysh.run ();
        }
        if (dimn == 3)
        {
            AdvectionDiffusion<3>  ysh (prm);
            ysh.run ();
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }

    return 0;
}
