//
//softpacking with PF 
//Aug 2017: development version of the code. 
//
#include "include/headers.h"
#include "parameters.h"
//physics files
#include "include/chemo.h"
#include "include/mechanics.h"

//Namespace
namespace softPacking
{
  using namespace dealii;

  //Initial conditions
  template <int dim>
  class InitalConditions: public Function<dim>{
  public:
    unsigned int dof;
    InitalConditions (unsigned int _dof): Function<dim>(totalDOF), dof(_dof){}
    void vector_value (const Point<dim>   &p, Vector<double>   &values) const{
      Assert (values.size() == totalDOF, ExcDimensionMismatch (values.size(), totalDOF));
      //
      for (unsigned int i=0;i <CDOFs; i++){
	values(2*i)=0.02 + 0.02*(0.5 -(double)(std::rand() % 100 )/100.0); //c
	values(2*i+1)=0.0; //mu
      }
      //initial seed cell
      if (p.distance(Point<dim>())<problemWidth/15){
	values(0)=0.99;
      }
    }
  };
  
  template <int dim>
  class multipleCH{
  public:
    multipleCH ();
    ~multipleCH ();
    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solveIteration ();
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle);
    double computeMass (const unsigned int cdof);
    void transferSolution (const unsigned int cdof);
    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;
    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;
    std::map<types::global_dof_index,Point<dim> > supportPoints;
    ConstraintMatrix                          constraints;
    LA::MPI::SparseMatrix                     system_matrix;
    LA::MPI::Vector                           locally_relevant_solution, U, Un, UGhost, UnGhost, dU;
    LA::MPI::Vector                           system_rhs;
    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;

    //solution variables
    unsigned int currentIncrement, currentIteration;
    double totalTime, currentTime, dt;
    std::vector<std::string> nodal_solution_names; std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;

    //cell division variables
    unsigned int nextAvailableField;
    std::vector<double> cellMassRatio;
    std::vector<double> cellCenter;
  };

  template <int dim>
  multipleCH<dim>::multipleCH ():
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    fe(FE_Q<dim>(FEOrder),totalDOF),
    dof_handler (triangulation),
    pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator)== 0)),
    computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times),
    cellMassRatio(CDOFs, 0.0), cellCenter(dim+1){
    //initial randon generator
    std::srand(1);
    
    //solution variables
    dt=TimeStep; totalTime=TotalTime;
    currentIncrement=0; currentTime=0;

    //set nextAvailableField
    nextAvailableField=1;
    
    //nodal solution names
    for (unsigned int i=0; i<CDOFs; ++i){
      char buffer[10];
      sprintf(buffer, "c%u", i+1); nodal_solution_names.push_back(buffer); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      sprintf(buffer, "mu%u", i+1); nodal_solution_names.push_back(buffer); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }
  }
  
  template <int dim>
  multipleCH<dim>::~multipleCH (){
    dof_handler.clear ();
  }

      //Setup
  template <int dim>
  void multipleCH<dim>::setup_system (){
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs (fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    DoFTools::map_dofs_to_support_points(MappingQ1<dim,dim>(), dof_handler, supportPoints);
    
    //Non-ghost vectors
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
    U.reinit (locally_owned_dofs, mpi_communicator);
    Un.reinit (locally_owned_dofs, mpi_communicator);
    dU.reinit (locally_owned_dofs, mpi_communicator);
    //Ghost vectors
    UGhost.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    UnGhost.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    /*
    //BC's
    std::vector<bool> uBC (totalDOF, false);
    for (unsigned int i=0; i<dim; i++){
      uBC[i]=true;
    }
    VectorTools::interpolate_boundary_values (dof_handler, 0, ZeroFunction<dim>(totalDOF), constraints, uBC);
    */
    constraints.close ();

    DynamicSparsityPattern dsp (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator, locally_relevant_dofs);
    system_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  }

  //Assembly
  template <int dim>
  void multipleCH<dim>::assemble_system (){
    TimerOutput::Scope t(computing_timer, "assembly");
    system_rhs=0.0; system_matrix=0.0;
    const QGauss<dim>  quadrature_formula(FEOrder+1);
    const QGauss<dim-1>	face_quadrature_formula (FEOrder+1);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_normal_vectors);
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    unsigned int n_q_points= fe_values.n_quadrature_points;
  
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned()){
	fe_values.reinit (cell);
	local_matrix = 0; local_rhs = 0; 
	cell->get_dof_indices (local_dof_indices);
	 //AD variables
	Table<1, Sacado::Fad::DFad<double> > ULocal(dofs_per_cell); Table<1, double > ULocalConv(dofs_per_cell);
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  if (std::abs(UGhost(local_dof_indices[i]))<1.0e-16){ULocal[i]=0.0;}
	  else{ULocal[i]=UGhost(local_dof_indices[i]);}
	  ULocal[i].diff (i, dofs_per_cell);
	  ULocalConv[i]= UnGhost(local_dof_indices[i]);
	}
	Table<1, Sacado::Fad::DFad<double> > R(dofs_per_cell); 
	for (unsigned int i=0; i<dofs_per_cell; ++i) {R[i]=0.0;}
	residualForChemo(fe_values, 0, fe_face_values, cell, dt, ULocal, ULocalConv, R, currentTime, totalTime, cellMassRatio, nextAvailableField);
	
	//Residual(R) and Jacobian(R')
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
	  for (unsigned int j=0; j<dofs_per_cell; ++j){
	    // R' by AD
	    local_matrix(i,j)= R[i].fastAccessDx(j);
	  }
	  //R
	  local_rhs(i) = -R[i].val();
	}
	constraints.distribute_local_to_global (local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
      }
    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
  }
  

  //Solve
  template <int dim>
  void multipleCH<dim>::solveIteration(){
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector completely_distributed_solution (locally_owned_dofs, mpi_communicator);
    /*    
    //Iterative solvers from Petsc and Trilinos
    SolverControl solver_control (dof_handler.n_dofs(), 1e-12);
#ifdef USE_PETSC_LA
    LA::SolverGMRES solver(solver_control, mpi_communicator);
#else
    LA::SolverGMRES solver(solver_control);
#endif
    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    //data.symmetric_operator = true;
#else
    // Trilinos defaults are good 
#endif
    preconditioner.initialize(system_matrix, data);
    solver.solve (system_matrix, completely_distributed_solution, system_rhs, preconditioner);
    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;
    */
    //Direct solver MUMPS
    SolverControl cn;
    PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
    solver.set_symmetric_mode(false);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);
    //
    constraints.distribute (completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
    dU = completely_distributed_solution; 
  }

  //Solve
  template <int dim>
  void multipleCH<dim>::solve(){
    double res=1, tol=1.0e-8, abs_tol=1.0e-14, initial_norm=0, current_norm=0;
    double machineEPS=1.0e-15;
    currentIteration=0;
    char buffer[200];
    while (true){
      if (currentIteration>=20){sprintf(buffer, "Maximum number of iterations reached without convergence. \n"); pcout<<buffer; break; exit (1);}
      if (current_norm>1/std::pow(tol,2)){sprintf(buffer, "\nNorm is too high. \n\n"); pcout<<buffer; break; exit (1);}
      assemble_system();
      current_norm=system_rhs.l2_norm();
      initial_norm=std::max(initial_norm, current_norm);
      res=current_norm/initial_norm;
      sprintf(buffer,"Inc:%3u (time:%10.3e, dt:%10.3e), Iter:%2u. Residual norm: %10.2e. Relative norm: %10.2e \n", currentIncrement, currentTime, dt,  currentIteration, current_norm, res); pcout<<buffer; 
      if ((currentIteration>1) && (res<tol || current_norm< abs_tol)){sprintf(buffer,"Residual converged in %u iterations.\n\n", currentIteration); pcout<<buffer; break;}
      solveIteration();
      U+=dU; UGhost=U; 
      ++currentIteration;
    }
    Un=U; UnGhost=Un;
  }

  
  //Error estimates
  template <int dim>
  void multipleCH<dim>::refine_grid (){
    TimerOutput::Scope t(computing_timer, "refine");
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(3),
                                        typename FunctionMap<dim>::type(),
                                        locally_relevant_solution,
                                        estimated_error_per_cell);
    parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number (triangulation,
				       estimated_error_per_cell,
				       0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
  }

  //Output
  template <int dim>
  void multipleCH<dim>::output_results (const unsigned int cycle) {
    TimerOutput::Scope t(computing_timer, "output");
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (UnGhost, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);    

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");
    
    data_out.build_patches ();
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
      std::vector<std::string> filenames;
      for (unsigned int i=0;
	   i<Utilities::MPI::n_mpi_processes(mpi_communicator);
	   ++i)
	filenames.push_back ("solution-" +
			     Utilities::int_to_string (cycle, 2) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output (("solution-" +
				    Utilities::int_to_string (cycle, 2) +
				    ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
  }

  //compute mass of each cell
  template <int dim>
  double multipleCH<dim>::computeMass (const unsigned int cdof){
    QGauss<dim>  quadrature(FEOrder+1);
    FEValues<dim> fe_values (fe, quadrature, update_values | update_JxW_values);
    const unsigned int n_q_points= fe_values.n_quadrature_points;
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    double cellMass= 0;
    std::vector<Vector<double> >   values;
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
      values.push_back(Vector<double>(2*CDOFs)); //fill the empty values vector with a Vector of size 2*CDOFs for values of each of the components
    }
    
    //loop over cells
    for (unsigned int i=0; i<dim+1; i++) cellCenter[i]=0.0;
    //
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell!=endc; ++cell){
      if (cell->is_locally_owned()){
	fe_values.reinit (cell);
	cell->get_dof_indices (local_dof_indices);
	fe_values.get_function_values(Un, values); //get the values for this cell
	//now loop over quadrature points in each cell
	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
	  MappingQ1<dim,dim> quadMap;
	  Point<dim> quadPoint(quadMap.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q_point)));
	  double cval=values[q_point][2*cdof];  
	  if (cval>=0.05){
	    cellMass += cval*fe_values.JxW(q_point);
	    for (unsigned int i=0; i<dim; i++) cellCenter[i]+=quadPoint[i]*cval;
	    cellCenter[dim]+=1;
	  }
	}
      }
    }
    //accumulate cell center data across all cores
    cellMass= Utilities::MPI::sum(cellMass, mpi_communicator);
    Utilities::MPI::sum(cellCenter, mpi_communicator, cellCenter);
    for (unsigned int i=0; i<dim; i++) cellCenter[i]/=cellCenter[dim];
    char buffer[100];
    sprintf(buffer, "cell %u is located at (%5.2e, %5.2e) with mass: %6.3e\n", cdof, cellCenter[0], cellCenter[1], cellMass); pcout << buffer;
    //
    return cellMass;
  }

  //post division, transfer half cell from one order parameter to next free order parameter
  template <int dim>
  void multipleCH<dim>::transferSolution (const unsigned int cdof){
    char buffer[100];
    sprintf(buffer, "***cell %u has doubled in mass. Dividing into cell %u and cell %u***\n", cdof, cdof, nextAvailableField); pcout << buffer;

    //get random angle for cell division (later be made an explicit function of the underlying mechanics and/or chemical signaling)
    const double pi = std::acos(-1);
    //double theta= (pi/180)*(std::rand() % 180);
    double theta=(pi/180)*45;
    if (nextAvailableField>=2) theta=(pi/180)*135;
    if (nextAvailableField>=4) theta=(pi/180)*45;
    if (nextAvailableField>=8) theta=(pi/180)*135;
    if (nextAvailableField>=16) theta=(pi/180)*45;
    sprintf(buffer, "division along %5.1f degree axis\n", 180*theta/pi); pcout << buffer;

    //split cell field into two half cell fields
    Point<dim> cellCenterPoint(cellCenter[0],cellCenter[1]);
    Point<dim> u(std::cos(theta), std::sin(theta)); //cell division axis
    
    //set the dof values of both vectors accordingly
    QGauss<dim>  quadrature(FEOrder+1);
    FEValues<dim> fe_values (fe, quadrature, update_values | update_JxW_values);
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    unsigned int c1, mu1, c2, mu2;
    std::vector<unsigned int> indexc1, indexmu1, indexc2, indexmu2;
    std::vector<double>       valuec1, valuemu1, valuec2, valuemu2;
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell!=endc; ++cell){
      if (cell->is_locally_owned()){
	fe_values.reinit (cell);
	cell->get_dof_indices (local_dof_indices);
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
	  const unsigned int id1 = fe_values.get_fe().system_to_component_index(i).first;
	  const unsigned int shapeID1= fe_values.get_fe().system_to_component_index(i).second;
	  //get id of mu1
	  if (id1==2*cdof){
	    c1=local_dof_indices[i];
	    bool foundmu1=false, foundc2=false, foundmu2=false;
	    for (unsigned int j=0; j<dofs_per_cell; ++j) {
	      if (fe_values.get_fe().system_to_component_index(j).second==shapeID1){
		const unsigned int id2 = fe_values.get_fe().system_to_component_index(j).first;
		if (id2==(2*cdof+1)) {mu1=local_dof_indices[j]; foundmu1=true;}
		if (id2==(2*nextAvailableField))   {c2 =local_dof_indices[j]; foundc2=true;}
		if (id2==(2*nextAvailableField+1)) {mu2=local_dof_indices[j]; foundmu2=true;}
	      }
	    }
	    if (!foundmu1 || !foundc2 || !foundmu2){std::cout << "ERROR: Couldnot find indices corresponding to mu1/c1/c2. \n"; exit(-1);}
	    //pcout << shapeID1 << " : " << c1 << " " << mu1 << " " << c2 << " " << mu2 << "\n";
	    //now find if nodal values should be switched
	    Point<dim> n1=supportPoints.find(local_dof_indices[i])->second;
	    Point<dim> v = n1; v-=cellCenterPoint; // vector connecting cell center to this node
	    double uXv= u[0]*v[1] - v[0]*u[1]; // Z component of u X v cross product
	    //copy points with +ve cross product (right side of cell division axis) onto new cell field
	    if (uXv>0.0){
	      /*if (locally_owned_dofs.is_element(c2)) {Un(c2)=Un(c1);}
	      if (locally_owned_dofs.is_element(mu2)) {Un(mu2)=Un(mu1);}
	      if (locally_owned_dofs.is_element(c1)) {Un(c1)=0.0;}
	      if (locally_owned_dofs.is_element(mu1)) {Un(mu1)=0.0;}
	      */
	      indexc1.push_back(c1); valuec1.push_back(0.02 + 0.02*(0.5 -(double)(std::rand() % 100 )/100.0));
	      indexmu1.push_back(mu1); valuemu1.push_back(0.0);
	      indexc2.push_back(c2); valuec2.push_back(Un(c1));
	      indexmu2.push_back(mu2); valuemu2.push_back(0.0); 
	    }
	  }
	}
      }
      //pcout << "\n";
    }
    Un.set(indexc1, valuec1);
    Un.set(indexmu1, valuemu1);
    Un.set(indexc2, valuec2);
    Un.set(indexmu2, valuemu2);
    Un.compress(VectorOperation::insert);
    U=Un; UnGhost=Un; nextAvailableField++;
  }
  
  //Run
  template <int dim>
  void multipleCH<dim>::run (){
    //setup problem geometry and mesh
#ifdef ellipticMesh
    GridGenerator::hyper_ball(triangulation, Point<dim>(), (double)problemWidth/2.0);
    static const HyperBallBoundary<dim> boundary(Point<dim>(), (double)problemWidth/2.0);
    triangulation.set_boundary (0, boundary);
    triangulation.refine_global (refinementFactor);
    //scale the geometry to make the circle into an ellipse
    std::set<unsigned int> tempSet;
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
    for (;cell!=endc; ++cell){
      if (cell->is_locally_owned()){
	for (unsigned int i=0; i<std::pow(2,dim); ++i){
	  if (tempSet.count(cell->vertex_index(i))== 0) {
	    cell->vertex(i)[1]=cell->vertex(i)[1]*ellipticityFactor;
	    tempSet.insert(cell->vertex_index(i));
	  }
	}
      }
    }
#else
    GridGenerator::hyper_cube (triangulation, -problemWidth/2.0, problemWidth/2.0);
    triangulation.refine_global (refinementFactor);
#endif
    setup_system ();
    pcout << "   Number of active cells:       "
	  << triangulation.n_global_active_cells()
	  << std::endl
	  << "   Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << std::endl;
    //scale the geometry to make the circle into an ellipse (still testing)
    /*
    std::set<unsigned int> tempSet;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (;cell!=endc; ++cell){
      for (unsigned int i=0; i<4; ++i){
	if (tempSet.count(cell->vertex_index(i))== 0) {
	  cell->vertex(i)[0]=cell->vertex(i)[0]*ELLIPSEFACTOR;
	  tempSet.insert(cell->vertex_index(i));
	}
      }
    }
    */
    
    //setup initial conditions
    VectorTools::interpolate(dof_handler, InitalConditions<dim>(0), U); Un=U;
    //Sync ghost vectors to non-ghost vectors
    UGhost=U;  UnGhost=Un;
    output_results (0);

    double initialCellMass= computeMass(0);
    //Time stepping
    currentIncrement=0;
    unsigned int counter=0; bool incCounter=false;
    for (currentTime=0; currentTime<=totalTime; currentTime+=dt){
      currentIncrement++;
      solve();
      //check for cell division
      for (unsigned int cells=0; cells<nextAvailableField; cells++){
	double cellMass=computeMass(cells); cellMassRatio[cells]=cellMass/initialCellMass;
	if ((cellMass>2*initialCellMass)){
	  if (nextAvailableField<CDOFs){
	    transferSolution(cells);
	    dt=TimeStep*1.0e-6; incCounter=true; counter=0;
	  }
	  else{
	    pcout << "should divide now, but no empty fields available. skipping division \n";
	  }
	}
      }
      
      output_results(currentIncrement);
      pcout << std::endl;
      if (incCounter){
	counter++;
	if ((counter>3)){
	  if (dt<TimeStep){
	    dt=TimeStep*1.0e-6*std::pow(counter-3,6);
	  }
	  else{
	    incCounter=false;
	  }
	}
      }
	
    }
    computing_timer.print_summary ();
  }
}


int main(int argc, char *argv[]){
  try
    {
      using namespace dealii;
      using namespace softPacking;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      multipleCH<2> problem;
      problem.run ();
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
