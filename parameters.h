//problem geometry, mesh control
#define DIMS 2
#define totalDOF (DIMS+2) //do not change this value
#define problemWidth 1.0
#define ELLIPSEFACTOR 1.0
#define refinementFactor 7

//helium properties
#define Source -1.0e3 //Helium production rate
#define InterfaceEnergyParameter {5.0e-3, 5.0e-3, 5.0e-3} //{Kx, Ky, Kz}: Parameters for isotropy of the interface energy
//#define InterfaceEnergyParameter {5.0e-3, 7.0e-3, 5.0e-3} //{Kx, Ky, Kz}: Parameters for anisotropy of the interface energy

//mechanics properties
#define elasticModulusC11 1.0e1
#define LatticeConstantChangeStrain 0.0
#define PressureValue 0.001 //Just for debugging, considering constant pressure

//phase field properties
#define PIval 3.14159265
#define dFdC  2*PIval*std::cos((4*c[q]-1.0)*PIval*0.5) //400*c[q]*(c[q]-1.0)*(c[q]-0.5)
#define Mobility 1.0


//time step controls
#define TimeStep 5.0e-6
#define TotalTime 100*TimeStep
#define timeForEquilibrium 10*TimeStep

//output controls
#define outputFileName "output"
