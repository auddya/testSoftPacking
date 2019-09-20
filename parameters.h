//problem geometry, mesh control
#define DIMS 2
#define FEOrder 1
#define problemWidth 1.0
#define ellipticMesh
#define ellipticityFactor 0.5
//#define bufferSpace problemWidth/20.0
#define refinementFactor 5

//order parameter controls
#define cellSize 100
#define CDOFs 5
#define totalDOF (2*CDOFs) 

//PF properties
#define Source 5.0e2 //species production rate
#define InterfaceEnergyParameter {1.0e-3, 1.0e-3, 1.0e-3} //{Kx, Ky, Kz}

//mechanics properties
#define elasticModulusC11 1.0e1
#define PressureValue 0.001 //Just for debugging, considering constant pressure

//phase field properties
#define dFdC  16*c[cDof][q]*(c[cDof][q]-1.0)*(c[cDof][q]-0.5) 
#define Mobility 1.0

//time step controls
#define TimeStep 5.0e-4
#define TotalTime 500*TimeStep
#define timeForEquilibrium 10*TimeStep

//output controls
#define outputFileName "output"
