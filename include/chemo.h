#ifndef CHEMO_H_
#define CHEMO_H_
#include "functionEvaluations.h"
#include "supplementaryFunctions.h"

//Mechanics residual implementation
template <int dim>
void residualForChemo(FEValues<dim>& fe_values, unsigned int DOF, FEFaceValues<dim>& fe_face_values, const typename DoFHandler<dim>::active_cell_iterator &cell, double dt, dealii::Table<1, Sacado::Fad::DFad<double> >& ULocal, dealii::Table<1, double>& ULocalConv, dealii::Table<1, Sacado::Fad::DFad<double> >& R, double currentTime, double totalTime, deformationMap<Sacado::Fad::DFad<double>, dim>& defMap){
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
	    if (c_conv[cDof][q]<0.99) {
	      //R[i] +=  -fe_values.shape_value(i, q)*Source*fe_values.JxW(q); //source term
	    }
	  }
	  for (unsigned int j = 0; j < dim; j++){
	    //With compositon dependent mobility M*c*(1-c)
	    //R[i] += fe_values.shape_value(i, q)*(M*(1-2*c[q])*c_j[q][j])*mu_j[q][j]*fe_values.JxW(q);
	    //R[i] += fe_values.shape_grad(i, q)[j]*(M*c[q]*(1-c[q]))*mu_j[q][j]*fe_values.JxW(q);
	    //With constant mobility
	    R[i] += fe_values.shape_grad(i, q)[j]*M*mu_j[cDof][q][j]*fe_values.JxW(q);
	  }
	}
	else if(ck==1){
	  Sacado::Fad::DFad<double> dfdc  = dFdC;
	  if (cDof==0) dfdc += 200*c[cDof][q]*c[1][q]*c[1][q];
	  else if (cDof==1) dfdc += 200*c[cDof][q]*c[0][q]*c[0][q];
	  R[i] +=  fe_values.shape_value(i, q)*(mu[cDof][q] - dfdc)*fe_values.JxW(q);
	  //R[i] +=  fe_values.shape_value(i, q)*(mu[q] - dfdc - defMap.W[q] - pressure*defMap.divU[q])*fe_values.JxW(q);
	  for (unsigned int j = 0; j < dim; j++){
	    Sacado::Fad::DFad<double> Kjj= Kappa[j];
	    //if (cDof==0) Kjj*= (1.0+0.01*c_j_norm[1][q]);
	    //else if (cDof==1) Kjj*= (1.0+0.01*c_j_norm[0][q]);
	    Sacado::Fad::DFad<double> kc_j= c_j[cDof][q][j]*Kjj; // Kjj*C_j
	    //Sacado::Fad::DFad<double> kc12_j=0.0;
	    //if (cDof==0) kc12_j= c_j[1][q][j]*Kjj*10; // Kjj*C_j
	    //if (cDof==1) kc12_j= c_j[0][q][j]*Kjj*10; // Kjj*C_j
	    R[i] -= fe_values.shape_grad(i, q)[j]*(kc_j)*fe_values.JxW(q);
	  }
	}
      }
    }
    //
  }
}

#endif /* CHEMO_H_ */
