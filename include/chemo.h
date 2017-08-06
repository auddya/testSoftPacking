#ifndef CHEMO_H_
#define CHEMO_H_
#include "functionEvaluations.h"
#include "supplementaryFunctions.h"

//Mechanics residual implementation
template <int dim>
void residualForChemo(FEValues<dim>& fe_values, unsigned int DOF, FEFaceValues<dim>& fe_face_values, const typename DoFHandler<dim>::active_cell_iterator &cell, double dt, dealii::Table<1, Sacado::Fad::DFad<double> >& ULocal, dealii::Table<1, double>& ULocalConv, dealii::Table<1, Sacado::Fad::DFad<double> >& R, double currentTime, double totalTime, const std::vector<double>& cellMassRatio, const unsigned int nextAvailableField){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;

  double pressure=PressureValue; //Just for debugging, considering constant pressure
  
  dealii::Table<2,Sacado::Fad::DFad<double> > c(CDOFs, n_q_points), mu(CDOFs, n_q_points);
  dealii::Table<3,Sacado::Fad::DFad<double> > c_j(CDOFs, n_q_points, dim);
  dealii::Table<3,Sacado::Fad::DFad<double> > mu_j(CDOFs, n_q_points, dim);
  dealii::Table<3,double> c_conv_j(CDOFs, n_q_points, dim);
  dealii::Table<2,double> c_conv(CDOFs, n_q_points);
  dealii::Table<2,Sacado::Fad::DFad<double> > c_j_norm(CDOFs, n_q_points);
  for (unsigned int cDof=0; cDof<CDOFs; cDof++){
    //evaluate gradients
    for (unsigned int q=0; q<n_q_points; ++q){
      c[cDof][q]=0.0; c_conv[cDof][q]=0; mu[cDof][q]=0.0;
      for (unsigned int j=0; j<dim; j++) {c_j[cDof][q][j]=0.0; c_conv_j[cDof][q][j]=0.0; mu_j[cDof][q][j]=0.0; }
      for (unsigned int i=0; i<dofs_per_cell; ++i) {
	const int ck = fe_values.get_fe().system_to_component_index(i).first - (DOF+2*cDof);
	if (ck==0) { c[cDof][q]+=fe_values.shape_value(i, q)*ULocal[i]; c_conv[cDof][q]+=fe_values.shape_value(i, q)*ULocalConv[i];}
	else if (ck==1) mu[cDof][q]+=fe_values.shape_value(i, q)*ULocal[i];
	for (unsigned int j=0; j<dim; j++) {
	  if (ck==0) {
	    c_j[cDof][q][j]+=fe_values.shape_grad(i, q)[j]*ULocal[i];
	    c_conv_j[cDof][q][j]+=fe_values.shape_grad(i, q)[j]*ULocalConv[i];
	  }
	  else if (ck==1) mu_j[cDof][q][j]+=fe_values.shape_grad(i, q)[j]*ULocal[i];
	}
      }
      c_j_norm[cDof][q]=0.0;
      for (unsigned int j=0; j<dim; j++){
	c_j_norm[cDof][q]+=c_j[cDof][q][j]*c_j[cDof][q][j];
      }
    }
  }
  

  //evaluate Residual
  for (unsigned int cDof=0; cDof<CDOFs; cDof++){
    double Kappa[] =InterfaceEnergyParameter;
    Sacado::Fad::DFad<double> M= Mobility;
    for (unsigned int i=0; i<dofs_per_cell; ++i) {
      const int ck = fe_values.get_fe().system_to_component_index(i).first - (DOF+2*cDof);
      for (unsigned int q=0; q<n_q_points; ++q){
	if (ck==0){
	  R[i] +=  (1/dt)*fe_values.shape_value(i, q)*(c[cDof][q]-c_conv[cDof][q])*fe_values.JxW(q);
	  if (currentTime>timeForEquilibrium){
	    if (c_conv[cDof][q]>0.99) {
	      double sourceValue=Source/2.0;
	      double ratio=cellMassRatio[cDof];
	      if (nextAvailableField<CDOFs){
		if (ratio>2.1) sourceValue*=0.0;
		else if (ratio<1.0) sourceValue*=2.0;
	      }
	      else{
		if (ratio>1.1) sourceValue*=0.0;
	      }
	      R[i] +=  -fe_values.shape_value(i, q)*sourceValue*fe_values.JxW(q); //source term
	    }
	  }
	  for (unsigned int j = 0; j < dim; j++){
	    //With constant mobility
	    R[i] += fe_values.shape_grad(i, q)[j]*M*mu_j[cDof][q][j]*fe_values.JxW(q);
	  }
	}
	else if(ck==1){
	  Sacado::Fad::DFad<double> dfdc  = dFdC;
	  //add cross penalty terms to free energy
	  for (unsigned int cDof2=0; cDof2<CDOFs; cDof2++){
	    if (cDof2!=cDof) dfdc += 200*c[cDof][q]*c[cDof2][q]*c[cDof2][q];
	  }
	  //add surface buffer zone
	  if (cell->at_boundary()){
	    dfdc += 200*c[cDof][q]*1.0*1.0;
	  }
	  /*
	  if (c[cDof][q].val()>0.1 || true){
	  MappingQ1<dim,dim> quadMap;
	  Point<dim> quadPoint(quadMap.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	  if (((quadPoint[0]<(-problemWidth/2+bufferSpace)) || (quadPoint[0]> (problemWidth/2-bufferSpace)) || (quadPoint[1]<(-problemWidth/2+bufferSpace)) || (quadPoint[1]>(problemWidth/2-bufferSpace)))){
	  dfdc += 200*c[cDof][q]*1.0*1.0;
	  }
	  }
	  */
	  //
	  R[i] +=  fe_values.shape_value(i, q)*(mu[cDof][q] - dfdc)*fe_values.JxW(q);
	  //R[i] +=  fe_values.shape_value(i, q)*(mu[q] - dfdc - defMap.W[q] - pressure*defMap.divU[q])*fe_values.JxW(q);
	  for (unsigned int j = 0; j < dim; j++){
	    Sacado::Fad::DFad<double> Kjj= Kappa[j];
	    Sacado::Fad::DFad<double> kc_j= c_j[cDof][q][j]*Kjj; // Kjj*C_j
	    R[i] -= fe_values.shape_grad(i, q)[j]*(kc_j)*fe_values.JxW(q);
	  }
	}
      }
    }
    //
  }
}

#endif /* CHEMO_H_ */
