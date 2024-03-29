#ifndef FUNCTIONEVALUATIONS_H_
#define FUNCTIONEVALUATIONS_H_
#include <deal.II/fe/fe_values.h>
using namespace dealii;

template <class T, int dim>
struct deformationMap{
deformationMap(unsigned int n_q_points): F(n_q_points, dim, dim),  invF(n_q_points, dim, dim), gradU(n_q_points, dim, dim), detF(n_q_points), divU(n_q_points), W(n_q_points){}
  Table<3, T> F, invF, gradU;
  Table<1, T> detF, divU, W;
};

template <class T, int dim>
void evaluateScalarFunction(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, Table<1, T>& U){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    U[q]=0.0; //U
    for (unsigned int k=0; k<dofs_per_cell; ++k){
      if (fe_values.get_fe().system_to_component_index(k).first==DOF){
	U[q]+=ULocal[k]*fe_values.shape_value(k, q); //U
      }
    }
  }
}

template <class T, int dim>
void evaluateScalarFunctionGradient(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, Table<2, T>& gradU, deformationMap<T, dim>& defMap, bool gradientInCurrentConfiguration){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  Table<1, T> refGradU(dim);
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    for (unsigned int i=0; i<dim; ++i){refGradU[i]=0.0;}
    for (unsigned int k=0; k<dofs_per_cell; ++k){
      if (fe_values.get_fe().system_to_component_index(k).first==DOF){
	for (unsigned int i=0; i<dim; ++i){
	  refGradU[i]+=ULocal[k]*fe_values.shape_grad(k, q)[i]; //gradU
	}
      }
    }
    //Transform gradient to current configuration. gradW=(F^-T)*GradW
    for (unsigned int i=0; i<dim; ++i){
      if (gradientInCurrentConfiguration==false) gradU[q][i]=refGradU[i];
      else{
	gradU[q][i]=0.0;
	for (unsigned int j=0; j<dim; ++j){
	  gradU[q][i]+=defMap.invF[q][j][i]*refGradU[j];
	}
      }
    }
  }
}

template <class T, int dim>
void evaluateVectorFunction(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, Table<2, T>& U){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    for (unsigned int i=0; i<dim; ++i){
      U[q][i]=0.0;
    }
    for (unsigned int k=0; k<dofs_per_cell; ++k){
      unsigned int ck = fe_values.get_fe().system_to_component_index(k).first - DOF;
      if (ck>=0 && ck<dim){
	U[q][ck]+=ULocal[k]*fe_values.shape_value(k, q); //U
      }
    }
  }
}

template <class T, int dim>
void evaluateVectorFunctionGradient(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, Table<3, T>& gradU){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;

  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	gradU[q][i][j]=0.0;
      }
    }
    for (unsigned int k=0; k<dofs_per_cell; ++k){
      unsigned int ck = fe_values.get_fe().system_to_component_index(k).first - DOF;
      if (ck>=0 && ck<dim){
	for (unsigned int i=0; i<dim; ++i){
	  gradU[q][ck][i]+=ULocal[k]*fe_values.shape_grad(k, q)[i]; //gradU
	}
      }
    }
  }
}

#endif /* FUNCTIONEVALUATIONS_H_ */

