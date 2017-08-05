#ifndef MECHANICS_H_
#define MECHANICS_H_
#include "functionEvaluations.h"
#include "supplementaryFunctions.h"

//Mechanics implementation
template <class T, int dim>
void getDeformationMap(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, deformationMap<T, dim>& defMap, const unsigned int iteration){
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //evaluate dx/dX
  Table<3, T> gradU(n_q_points, dim, dim);
  evaluateVectorFunctionGradient<T, dim>(fe_values, DOF, ULocal, gradU);
  
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    Table<2, T > Fq(dim, dim), invFq(dim, dim); T detFq;
    defMap.divU[q]=0.0;       
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.F[q][i][j] = Fq[i][j] = (i==j) + gradU[q][i][j]; //F (as double value)
	defMap.gradU[q][i][j] = gradU[q][i][j]; 
      }
      defMap.divU[q] += gradU[q][i][i];       
    }
    getInverse<T, dim>(Fq, invFq, detFq); //get inverse(F)
    defMap.detF[q] = detFq;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.invF[q][i][j] = invFq[i][j];
      }
    }
    //detF
    if (defMap.detF[q].val()<=1.0e-15 && iteration==0){
      printf("**************Non positive jacobian detected**************. Value: %12.4e\n", defMap.detF[q].val());
      for (unsigned int i=0; i<dim; ++i){
	for (unsigned int j=0; j<dim; ++j) printf("%12.6e  ", defMap.F[q][i][j].val());
	printf("\n"); exit(-1);
      }
      //throw "Non positive jacobian detected";
    }

    //compute strain energy density, W
    //E
    Table<2, Sacado::Fad::DFad<double> > E (dim, dim);
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	//small strain: E is epsilon
	E[i][j] = 0.5*(defMap.gradU[q][i][j]+defMap.gradU[q][j][i]);
      }
    }
    //S
    Table<2, Sacado::Fad::DFad<double> > S (dim, dim);
    double C11=elasticModulusC11;
    double C12=C11/2.0, C44=C11/2.5;
    if(dim==2){
      S[0][0]=C11*E[0][0]+C12*E[1][1];
      S[1][1]=C12*E[0][0]+C11*E[1][1];
      S[0][1]=S[1][0]=C44*E[0][1];
    }
    //W
    defMap.W[q]=0.0;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.W[q]+=0.5*S[i][j]*E[i][j];
      }
    }
  }
}

//Mechanics implementation
template <class T, int dim>
  void evaluateStress(const FEValues<dim>& fe_values,const unsigned int DOF, const Table<1, T>& ULocal, Table<3, T>& P, const deformationMap<T, dim>& defMap, typename DoFHandler<dim>::active_cell_iterator& cell){
  unsigned int n_q_points= fe_values.n_quadrature_points;
  
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    //Fe
    Table<2, Sacado::Fad::DFad<double> > Fe (dim, dim);
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Fe[i][j]=defMap.F[q][i][j];
      }
    }
    //E
    double Ec=0.0; //c_conv[q]*LatticeConstantChangeStrain;
    Table<2, Sacado::Fad::DFad<double> > E (dim, dim);
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	/*
	//finite strain
	E[i][j] = -0.5*(i==j);
	for (unsigned int k=0; k<dim; ++k){
	  E[i][j] += 0.5*Fe[k][i]*Fe[k][j];
	}
	*/
	//small strain: E is epsilon
	E[i][j] = 0.5*(defMap.gradU[q][i][j]+defMap.gradU[q][j][i]);
      }
      //E[i][i]+=Ec;
    }
    //S
    Table<2, Sacado::Fad::DFad<double> > S (dim, dim);
    double C11=elasticModulusC11;
    double C12=C11/2.0, C44=C11/2.5;
    if(dim==2){
      S[0][0]=C11*E[0][0]+C12*E[1][1];
      S[1][1]=C12*E[0][0]+C11*E[1][1];
      S[0][1]=S[1][0]=C44*E[0][1];
    }
    else throw "dim not equal to 2";
    //P
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	//small strain: P is sigma
	P[q][i][j]=S[i][j];
	/*
	//finite strain
	P[q][i][j]=0;
	for (unsigned int k=0; k<dim; ++k){
	  P[q][i][j]+=Fe[i][k]*S[k][j];
	}
	*/
      }
    }
    /*
    Table<2, Sacado::Fad::DFad<double> > Sigma (dim, dim);
    //Sigma
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Sigma[i][j]=0;
	for (unsigned int k=0; k<dim; ++k){
	  Sigma[i][j]+=P[q][i][k]*defMap.F[q][j][k];
	}
	Sigma[i][j]/=defMap.detF[q];
      }
    }
    */
  }
}

//Mechanics residual implementation
template <int dim>
void residualForMechanics(FEValues<dim>& fe_values, unsigned int DOF, Table<1, Sacado::Fad::DFad<double> >& ULocal, Table<1, double>& ULocalConv, Table<1, Sacado::Fad::DFad<double> >& R, deformationMap<Sacado::Fad::DFad<double>, dim>& defMap, typename DoFHandler<dim>::active_cell_iterator& cell){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;

  double pressure=0.0; //PressureValue; //Just for debugging, considering constant pressure
  
  //Temporary arrays
  Table<3,Sacado::Fad::DFad<double> > P (n_q_points, dim, dim);
  //evaluate mechanics
  evaluateStress<Sacado::Fad::DFad<double>, dim>(fe_values, DOF, ULocal, P, defMap, cell);
  
  //evaluate Residual
  for (unsigned int i=0; i<dofs_per_cell; ++i) {
    const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
    if (ck>=0 && ck<dim){
      // R = Grad(w)*P
      for (unsigned int q=0; q<n_q_points; ++q){
	double g=1.0; //c_conv[q]; //Considering g=eta
	for (unsigned int d = 0; d < dim; d++){
	  //R[i] +=  fe_values.shape_grad(i, q)[d]*P[q][ck][d]*fe_values.JxW(q);
	  R[i] +=  fe_values.shape_grad(i, q)[d]*g*P[q][ck][d]*fe_values.JxW(q);
	}
	R[i] +=  fe_values.shape_grad(i, q)[ck]*(g*pressure)*fe_values.JxW(q);
      }
    }
  }
}

#endif /* MECHANICS_H_ */
