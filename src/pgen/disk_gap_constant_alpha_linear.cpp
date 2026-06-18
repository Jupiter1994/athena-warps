//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp" 
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"


#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
Real vRProfileCyl(const Real rad, const Real phi, const Real z);
Real alphaProfile(const Real rad, const Real phi, const Real z);
void Load_ur1_Table(const std::string& fname);
Real ur1_from_Table(Real r);
void Load_alph_Table(const std::string& fname);
Real alph_from_Table(Real r);
void Load_beta_Table(const std::string& fname);
Real beta_from_Table(Real r);
void Load_gamm_Table(const std::string& fname);
Real gamm_from_Table(Real r);
// problem parameters which are useful to make global to this file
Real gm0, r0;
// for the density profile
Real Sigma0, dslope, r_1, r_2, xi_1, xi_2, dfloor;
// for the sound speed profile
Real p0_over_r0, pslope, gamma_gas;
Real Omega0;
// for W
Real R_in, R_out, W_in, W_out, dWdt, dpdt, time_fix;
std::vector<Real> ur1_table_r;
std::vector<Real> ur1_table_val;
std::vector<Real> alph_table_r;
std::vector<Real> alph_table_val;
std::vector<Real> beta_table_r;
std::vector<Real> beta_table_val;
std::vector<Real> gamm_table_r;
std::vector<Real> gamm_table_val;
std::string ur1_file, alph_file, beta_file, gamm_file;
// for visocisty
Real alpha_0, alpha_gap, r_gap_a, r_gap_b, del_a, del_b, depth_gap;
// for AMR
Real rho_AMR_limit;
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void alpha_viscosity(HydroDiffusion *phdif, MeshBlock *pmb, 
              const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc, 
              int is, int ie, int js, int je,int ks, int ke);
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar);
int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",1.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density 
  Sigma0 = pin->GetOrAddReal("problem","Sigma0",1.0);
  dslope = pin->GetOrAddReal("problem","dslope",0.0);
  r_1    = pin->GetOrAddReal("problem","r_1",1.0);
  r_2    = pin->GetOrAddReal("problem","r_2",1.0);
  xi_1   = pin->GetOrAddReal("problem","xi_1",2.0);
  xi_2   = pin->GetOrAddReal("problem","xi_2",2.0);

  // Get parameters for alpha viscosity
  alpha_0   = pin->GetOrAddReal("problem","alpha_0",0.0);
  alpha_gap = pin->GetOrAddReal("problem","alpha_gap",alpha_0);
  r_gap_a   = pin->GetOrAddReal("problem","r_gap_a",1.5);
  r_gap_b   = pin->GetOrAddReal("problem","r_gap_b",2.0);
  del_a     = pin->GetOrAddReal("problem","del_a",0.1);
  del_b     = pin->GetOrAddReal("problem","del_b",0.1);
  depth_gap = pin->GetOrAddReal("problem","depth_gap",3.0);

  // Get parameters for initial W
  R_in  = pin->GetReal("mesh","x1min");
  R_out = pin->GetReal("mesh","x1max");
  W_in  = pin->GetOrAddReal("problem","W_in",0.0);
  W_out = pin->GetOrAddReal("problem","W_out",0.0);
  ur1_file  = pin->GetOrAddString("problem", "ur1_profile_file", "ur1_profile_var_alpha.dat");
  alph_file = pin->GetOrAddString("problem", "alph_profile_file", "alph_profile_var_alpha.dat");
  beta_file = pin->GetOrAddString("problem", "beta_profile_file", "beta_profile_var_alpha.dat");
  gamm_file = pin->GetOrAddString("problem", "gamm_profile_file", "gamm_profile_var_alpha.dat");

  // Get parameters for disk stretching
  dWdt  = pin->GetOrAddReal("problem","dWdt",0.0);
  dpdt  = pin->GetOrAddReal("problem","dpdt",0.0);
  time_fix = pin->GetOrAddReal("problem","time_fix",0.0);

  // Get parameters for AMR
  rho_AMR_limit  = pin->GetOrAddReal("problem","rho_AMR_limit",0.1);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope     = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas  = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro","iso_sound_speed"));
    pslope     = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas  = pin->GetReal("hydro","gamma");
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  // enroll cooling
  if (NON_BAROTROPIC_EOS) {
    EnrollUserExplicitSourceFunction(Cooling);
  }  

  // enroll alpha viscosity
  if (alpha_0>1e-8){
    EnrollViscosityCoefficient(alpha_viscosity); // alpha viscosity
  }

  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real rad_W(0.0), phi_W(0.0), z_W(0.0);
  Real x(0.0), y(0.0);
  Real x_W(0.0), y_W(0.0), W;
  Real den_W, vel_W, vel_W_rad, vel_Wx, vel_Wy, vel_Wz;
  Real den, vel_x, vel_y, vel_z, vel_R, vel_phi;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  Load_ur1_Table(ur1_file);  // or get filename from input parameters
  Load_alph_Table(alph_file);  // or get filename from input parameters
  Load_beta_Table(beta_file);  // or get filename from input parameters
  Load_gamm_Table(gamm_file);  // or get filename from input parameters
  //if (Globals::my_rank == 0) {
  //  std::cout << "Loaded ur1 table from " << ur1_file << std::endl;
  //  std::cout << "Loaded alph table from " << alph_file << std::endl;
  //  std::cout << "Loaded beta table from " << beta_file << std::endl;
  //  std::cout << "Loaded gamm table from " << gamm_file << std::endl;
  //}

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);

        // get the  coordinates
        GetCylCoord(pcoord,rad,phi,z,i,j,k); 
        x = rad*cos(phi);
        y = rad*sin(phi);

        // apply rotation 
        W = (W_out-W_in)/(R_out-R_in) * (std::sqrt(rad*rad+z*z)-R_in) + W_in;
        Real ur1_coef  = ur1_from_Table(std::sqrt(rad*rad + z*z))*W_out*0;
        Real alph_rotation = alph_from_Table(std::sqrt(rad*rad + z*z))*0;
        Real beta_rotation = beta_from_Table(std::sqrt(rad*rad + z*z))*W_out*0 + W;
        Real gamm_rotation = gamm_from_Table(std::sqrt(rad*rad + z*z))*0;

        Real xp = +x*cos(gamm_rotation) + y*sin(gamm_rotation);
        Real yp = -x*sin(gamm_rotation) + y*cos(gamm_rotation);
        Real zp = z;

        Real xpp = +xp*cos(beta_rotation) - zp*sin(beta_rotation);
        Real ypp = yp;
        Real zpp = +xp*sin(beta_rotation) + zp*cos(beta_rotation);

        x_W = +xpp*cos(alph_rotation) + ypp*sin(alph_rotation);
        y_W = -xpp*sin(alph_rotation) + ypp*cos(alph_rotation);
        z_W = zpp;        
        rad_W = std::sqrt(x_W*x_W+y_W*y_W);
        phi_W = std::atan2(y_W,x_W);

        //rad_W = x1;
        //tha_W = atan2(std::sqrt(x_W*x_W+y_W*y_W),z_W);
        //phi_W = atan2(y_W,x_W);


        // compute initial conditions in cylindrical coordinates

        // background
        den_W = DenProfileCyl(rad_W,phi_W,z_W); //DenProfileCyl(rad_W,phi_W,0.0);
        vel_W = VelProfileCyl(rad_W,phi_W,z_W);
        

        Real vr1 = ur1_coef*vel_W*cos(phi_W)*z_W/std::sqrt(p0_over_r0/(gm0/rad_W/rad_W/rad_W));
        vel_W_rad = vr1 * rad_W/std::sqrt(rad*rad+z_W*z_W);

        den = den_W; // no need to further rotate scalar
        den = std::max(den,dfloor);

        vel_Wx = cos(phi_W)*vel_W_rad - sin(phi_W)*vel_W;
        vel_Wy = sin(phi_W)*vel_W_rad + cos(phi_W)*vel_W;
        vel_Wz = 0; //vr1 * z_W/std::sqrt(rad*rad+z_W*z_W);

        //vel_Wx = sin(tha_W)*cos(phi_W)*vel_W_rad - sin(phi_W)*vel_W;
        //vel_Wy = sin(tha_W)*sin(phi_W)*vel_W_rad + cos(phi_W)*vel_W;
        //vel_Wz = cos(tha_W)*vel_W_rad;       

        Real vel_xp = vel_Wx*cos(alph_rotation) - vel_Wy*sin(alph_rotation);
        Real vel_yp = vel_Wx*sin(alph_rotation) + vel_Wy*cos(alph_rotation);
        Real vel_zp = vel_Wz;

        Real vel_xpp = vel_xp*cos(beta_rotation) + vel_zp*sin(beta_rotation);
        Real vel_ypp = vel_yp;
        Real vel_zpp = -vel_xp*sin(beta_rotation) + vel_zp*cos(beta_rotation);

        vel_x = vel_xpp*cos(gamm_rotation) - vel_ypp*sin(gamm_rotation);
        vel_y = vel_xpp*sin(gamm_rotation) + vel_ypp*cos(gamm_rotation);
        vel_z = vel_zpp;

        vel_R   = (+x*vel_x+y*vel_y)/rad;
        vel_phi = (-y*vel_x+x*vel_y)/rad;

        if (porb->orbital_advection_defined)
          vel_phi -= vK(porb, x1, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IDN,k,j,i) = den;
          phydro->u(IM1,k,j,i) = den*vel_R;
          phydro->u(IM2,k,j,i) = den*vel_phi;
          phydro->u(IM3,k,j,i) = den*vel_z;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          Real r_polar = std::sqrt(rad*rad+z*z);
          phydro->u(IDN,k,j,i) = den;
          phydro->u(IM1,k,j,i) = den * (rad*vel_R + z*vel_z)/r_polar;
          phydro->u(IM2,k,j,i) = den * (z*vel_R - rad*vel_z)/r_polar;
          phydro->u(IM3,k,j,i) = den*vel_phi;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! alpha viscosity

void alpha_viscosity(HydroDiffusion *phdif, MeshBlock *pmb, 
              const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc, 
              int is, int ie, int js, int je,int ks, int ke){
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs2, vK, alpha_R, nu_v;
  //for (int k = pmb->ks; k <= pmb->ke; ++k) {
  for (int k = pmb->ks-2; k <= pmb->ke+2; ++k) {
    for (int j = pmb->js-2; j <= pmb->je+2; ++j) {
#pragma omp simd
      for (int i = pmb->is-2; i <= pmb->ie+2; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); 
        rad   = std::sqrt(rad*rad+z*z);
        cs2   = p0_over_r0 * std::pow(rad, pslope); // * (1 + 0.5*pslope*std::pow(z/rad,2));
        vK    = std::sqrt(gm0/rad);
        alpha_R = alphaProfile(0.5*(r_gap_a+r_gap_b),phi,z);
        nu_v  = alpha_R* cs2 / (vK/rad);
        //nu_v  = alpha_0* cs2 / (vK/rad);
        //printf("%1.9f \n",nu_v);
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = nu_v;
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! Locally isothermal cooling

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  Real g = pmb->peos->GetGamma();
  Real rad(0.0), phi(0.0), z(0.0);
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); 
        Real p_over_r = PoverR(rad,phi,z);
        cons(IEN,k,j,i) = prim(IDN,k,j,i)*p_over_r/(g-1.0);
        cons(IEN,k,j,i) += 0.5*prim(IDN,k,j,i)*(SQR(prim(IVX,k,j,i))+SQR(prim(IVY,k,j,i))+SQR(prim(IVZ,k,j,i)));
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}
 
    
//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
} 

//----------------------------------------------------------------------------------------
//! helper: beta and gamma

// Load ur1(r) from file
void Load_ur1_Table(const std::string& fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    std::cerr << "Failed to open ur1 table file: " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real r, ur1;
  while (file >> r >> ur1) {
    ur1_table_r.push_back(r);
    ur1_table_val.push_back(ur1);
  }
  file.close();
}

// Linear interpolation for ur1(r)
Real ur1_from_Table(Real r) {
  int n = ur1_table_r.size();
  if (r <= ur1_table_r.front()) return ur1_table_val.front();
  if (r >= ur1_table_r.back())  return ur1_table_val.back();

  for (int i = 0; i < n - 1; ++i) {
    if (r >= ur1_table_r[i] && r <= ur1_table_r[i + 1]) {
      Real r1 = ur1_table_r[i], r2 = ur1_table_r[i + 1];
      Real ur11 = ur1_table_val[i], ur12 = ur1_table_val[i + 1];
      return ur11 + (ur12 - ur11) * (r - r1) / (r2 - r1);
    }
  }
  return ur1_table_val.back(); // should not reach here
}

// Load alpha(r) from file
void Load_alph_Table(const std::string& fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    std::cerr << "Failed to open alpha table file: " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real r, alph;
  while (file >> r >> alph) {
    alph_table_r.push_back(r);
    alph_table_val.push_back(alph);
  }
  file.close();
}

// Linear interpolation for beta(r)
Real alph_from_Table(Real r) {
  int n = alph_table_r.size();
  if (r <= alph_table_r.front()) return alph_table_val.front();
  if (r >= alph_table_r.back())  return alph_table_val.back();

  for (int i = 0; i < n - 1; ++i) {
    if (r >= alph_table_r[i] && r <= alph_table_r[i + 1]) {
      Real r1 = alph_table_r[i], r2 = alph_table_r[i + 1];
      Real alph1 = alph_table_val[i], alph2 = alph_table_val[i + 1];
      return alph1 + (alph2 - alph1) * (r - r1) / (r2 - r1);
    }
  }
  return alph_table_val.back(); // should not reach here
}

// Load beta(r) from file
void Load_beta_Table(const std::string& fname) {
    std::ifstream file(fname);
    if (!file.is_open()) {
      std::cerr << "Failed to open beta table file: " << fname << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real r, beta;
    while (file >> r >> beta) {
      beta_table_r.push_back(r);
      beta_table_val.push_back(beta);
    }
    file.close();
}
  
// Linear interpolation for beta(r)
Real beta_from_Table(Real r) {
    int n = beta_table_r.size();
    if (r <= beta_table_r.front()) return beta_table_val.front();
    if (r >= beta_table_r.back())  return beta_table_val.back();
  
    for (int i = 0; i < n - 1; ++i) {
      if (r >= beta_table_r[i] && r <= beta_table_r[i + 1]) {
        Real r1 = beta_table_r[i], r2 = beta_table_r[i + 1];
        Real beta1 = beta_table_val[i], beta2 = beta_table_val[i + 1];
        return beta1 + (beta2 - beta1) * (r - r1) / (r2 - r1);
      }
    }
    return beta_table_val.back(); // should not reach here
}

// Load gamm(r) from file
void Load_gamm_Table(const std::string& fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    std::cerr << "Failed to open gamm table file: " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real r, beta;
  while (file >> r >> beta) {
    gamm_table_r.push_back(r);
    gamm_table_val.push_back(beta);
  }
  file.close();
}

// Linear interpolation for beta(r)
Real gamm_from_Table(Real r) {
  int n = gamm_table_r.size();
  if (r <= gamm_table_r.front()) return gamm_table_val.front();
  if (r >= gamm_table_r.back())  return gamm_table_val.back();

  for (int i = 0; i < n - 1; ++i) {
    if (r >= gamm_table_r[i] && r <= gamm_table_r[i + 1]) {
      Real r1 = gamm_table_r[i], r2 = gamm_table_r[i + 1];
      Real gamm1 = gamm_table_val[i], gamm2 = gamm_table_val[i + 1];
      return gamm1 + (gamm2 - gamm1) * (r - r1) / (r2 - r1);
    }
  }
  return gamm_table_val.back(); // should not reach here
}

//----------------------------------------------------------------------------------------
//! background state helpers: Sigma's

Real get_Sigma(const Real rad, const Real phi, const Real z) {
  Real taper_inner = exp(-std::pow(rad / r_1, xi_1));
  Real taper_outer = exp(-std::pow(rad / r_2, xi_2));
  Real main_term   = std::pow(rad, dslope);

  Real den = Sigma0 * taper_inner * main_term * taper_outer;
  //return std::max(den,dfloor);
  return den;
}

Real get_dSigma_dr(const Real rad, const Real phi, const Real z) {
  Real taper_inner = exp(-std::pow(rad / r_1, xi_1));
  Real taper_outer = exp(-std::pow(rad / r_2, xi_2));
  Real main_term   = std::pow(rad, dslope);

  Real d_taper_inner_dr = -xi_1 * std::pow(rad / r_1, xi_1 - 1) / r_1 * taper_inner;
  Real d_taper_outer_dr = -xi_2 * std::pow(rad / r_2, xi_2 - 1) / r_2 * taper_outer;
  Real d_main_term_dr = dslope * std::pow(rad, dslope - 1);

  return Sigma0 * (
      d_taper_inner_dr * main_term * taper_outer +
      taper_inner * d_main_term_dr * taper_outer +
      taper_inner * main_term * d_taper_outer_dr
    );
}

Real get_d2Sigma_dr2(const Real rad, const Real phi, const Real z) {

  Real taper_inner = exp(-std::pow(rad / r_1, xi_1));
  Real taper_outer = exp(-std::pow(rad / r_2, xi_2));
  Real main_term   = pow(rad, dslope);

  Real d_taper_inner_dr = -xi_1 * std::pow(rad / r_1, xi_1 - 1) / r_1 * taper_inner;
  Real d_taper_outer_dr = -xi_2 * std::pow(rad / r_2, xi_2 - 1) / r_2 * taper_outer;
  Real d_main_term_dr = dslope * std::pow(rad, dslope - 1);

  Real d2_taper_inner_dr2 = (-xi_1 * std::pow(rad / r_1, xi_1 - 1) / r_1 * d_taper_inner_dr
                                 - xi_1 * (xi_1 - 1) * std::pow(rad / r_1, xi_1 - 2) / r_1 * taper_inner);
  Real d2_taper_outer_dr2 = (-xi_2 * std::pow(rad / r_2, xi_2 - 1) / r_2 * d_taper_outer_dr
                                 - xi_2 * (xi_2 - 1) * std::pow(rad / r_2, xi_2 - 2) / r_2 * taper_outer);
  Real d2_main_term_dr2 = dslope * (dslope - 1) * std::pow(rad, dslope - 2);

  return Sigma0 * (
      d2_taper_inner_dr2 * main_term * taper_outer +
      d_taper_inner_dr * d_main_term_dr * taper_outer +
      d_taper_inner_dr * main_term * d_taper_outer_dr +
      taper_inner * d2_main_term_dr2 * taper_outer +
      taper_inner * d_main_term_dr * d_taper_outer_dr +
      taper_inner * main_term * d2_taper_outer_dr2
  );
}

//----------------------------------------------------------------------------------------
//! background state helpers: cs2's

Real get_cs2(const Real rad, const Real phi, const Real z) {
  return p0_over_r0 * std::pow(rad, pslope) * (1 + 0.5*pslope*std::pow(z/rad,2));
  //Real r2 = rad*rad+z*z;
  //return p0_over_r0 * std::pow(r2, 0.5*pslope);
}

Real get_dcs2_dr(const Real rad, const Real phi, const Real z) {
  return p0_over_r0 * pslope * std::pow(rad, pslope-1) * (1 + 0.5*pslope*std::pow(z/rad,2));
  //Real r2 = rad*rad+z*z;
  //return p0_over_r0 * pslope * rad * std::pow(r2, 0.5*pslope-1);
}

Real get_d2cs2_dr2(const Real rad, const Real phi, const Real z) {
  return p0_over_r0 * pslope * (pslope-1) * std::pow(rad, pslope-2) * (1 + 0.5*pslope*std::pow(z/rad,2));
  //Real r2 = rad*rad+z*z;
  //return p0_over_r0 * (pslope*std::pow(r2, 0.5*pslope-1) + pslope*(pslope-2)*rad*rad*std::pow(r2,0.5*pslope-2));
}

//----------------------------------------------------------------------------------------
//! background state helpers: H's

Real get_H(const Real rad, const Real phi, const Real z) {
  Real cs = std::sqrt(get_cs2(rad,phi,0.0));
  Real vK = std::pow(rad, -0.5);
  return cs / vK * rad;
}

Real get_dH_dr(const Real rad, const Real phi, const Real z) {
  Real cs2       = get_cs2(rad,phi,0.0);
  Real cs        = std::sqrt(cs2);
  Real inv_omega = std::pow(rad, 1.5);

  Real dcs_dr = 0.5 * get_dcs2_dr(rad,phi,0.0)/cs;
  Real dinv_omega_dr = 1.5 * std::pow(rad, 0.5);
  
  return dcs_dr * inv_omega + cs * dinv_omega_dr;
}

Real get_d2H_dr2(const Real rad, const Real phi, const Real z) {
  Real cs2       = get_cs2(rad,phi,0.0);
  Real cs        = std::sqrt(cs2);
  Real inv_omega = std::pow(rad, 1.5);

  Real dcs_dr = 0.5 * get_dcs2_dr(rad,phi,0.0)/cs;
  Real dinv_omega_dr = 1.5 * std::pow(rad, 0.5);
  
  Real d2cs_dr2 = 0.5 * (get_d2cs2_dr2(rad,phi,0.0)*cs - get_dcs2_dr(rad,phi,0.0)*dcs_dr)/cs2;
  Real d2inv_omega_dr2 = 0.5 * 1.5 * std::pow(rad, -0.5);

  return d2cs_dr2 * inv_omega + 2 * dcs_dr * dinv_omega_dr + cs * d2inv_omega_dr2;
}

//----------------------------------------------------------------------------------------
//! background state helpers: rho's

Real get_rho_mid(const Real rad, const Real phi, const Real z) {
  return get_Sigma(rad,phi,0.0)/get_H(rad,phi,0.0)*0.3989422804; // 1/sqrt(2pi)
}

Real get_drho_mid_dr(const Real rad, const Real phi, const Real z) {
  Real Sigma     = get_Sigma(rad,phi,0.0);
  Real dSigma_dr = get_dSigma_dr(rad,phi,0.0);
  Real Hdisk     = get_H(rad,phi,0.0);
  Real dH_dr     = get_dH_dr(rad,phi,0.0);
  return 0.3989422804 * (dSigma_dr*Hdisk - Sigma*dH_dr)/Hdisk/Hdisk;
}

Real get_d2rho_mid_dr2(const Real rad, const Real phi, const Real z) {
  Real Sigma       = get_Sigma(rad,phi,0.0);
  Real dSigma_dr   = get_dSigma_dr(rad,phi,0.0);
  Real d2Sigma_dr2 = get_d2Sigma_dr2(rad,phi,0.0);
  Real Hdisk       = get_H(rad,phi,0.0);
  Real dH_dr       = get_dH_dr(rad,phi,0.0);
  Real d2H_dr2     = get_d2H_dr2(rad,phi,0.0);

  Real term1 = +(d2Sigma_dr2*Hdisk-Sigma*d2H_dr2)/Hdisk/Hdisk;
  Real term2 = -(dSigma_dr*Hdisk-Sigma*dH_dr)*2.0/Hdisk/Hdisk/Hdisk*dH_dr;
  return 0.3989422804 * (term1 + term2);
}

//----------------------------------------------------------------------------------------
//! background state helpers: P's

Real get_P_mid(const Real rad, const Real phi, const Real z) {
  return get_cs2(rad,phi,0.0) * get_rho_mid(rad,phi,0.0);
}

Real get_dP_mid_dr(const Real rad, const Real phi, const Real z) {
  Real rho_mid = get_rho_mid(rad,phi,0.0);
  Real drho_dr = get_drho_mid_dr(rad,phi,0.0);
  Real cs2_mid = get_cs2(rad,phi,0.0);
  Real dcs2_dr = get_dcs2_dr(rad,phi,0.0);
  return dcs2_dr*rho_mid + cs2_mid*drho_dr;
}

Real get_d2P_mid_dr2(const Real rad, const Real phi, const Real z) {
  Real rho_mid   = get_rho_mid(rad,phi,0.0);
  Real drho_dr   = get_drho_mid_dr(rad,phi,0.0);
  Real d2rho_dr2 = get_d2rho_mid_dr2(rad,phi,0.0);
  Real cs2_mid   = get_cs2(rad,phi,0.0);
  Real dcs2_dr   = get_dcs2_dr(rad,phi,0.0);
  Real d2cs2_dr2 = get_d2cs2_dr2(rad,phi,0.0);
  return d2cs2_dr2*rho_mid + 2*dcs2_dr*drho_dr + cs2_mid*d2rho_dr2;
}

//----------------------------------------------------------------------------------------
//! background state helpers: Viscosity's

Real gap_profile(const Real rad, const Real phi, const Real z) {
    Real r_polar = std::sqrt(rad*rad+z*z);
    Real gap_func = std::tanh((r_polar-r_gap_a)/del_a) - std::tanh((r_polar-r_gap_b)/del_b);
    return 1 + 0.5*(depth_gap - 1.0) * gap_func;
}

Real alphaProfile(const Real rad, const Real phi, const Real z) {
    Real gap_func = gap_profile(rad,phi,z);
    return alpha_0 * gap_func;
}

Real dlnalpha_dlnR(const Real rad, const Real phi, const Real z) {
    if (std::abs(depth_gap-1.0)<1e-5){
        return 0.0;
    }
    Real r_polar = std::sqrt(rad*rad+z*z);
    Real x1 = (r_polar-r_gap_a)/del_a;
    Real x2 = (r_polar-r_gap_b)/del_b;

    Real sech_term_a = pow(std::cosh((r_gap_a-r_polar)/del_a),-2)/del_a;
    Real sech_term_b = pow(std::cosh((r_gap_b-r_polar)/del_b),-2)/del_b;
    Real tanh_term   = std::tanh((r_polar-r_gap_a)/del_a) - std::tanh((r_polar-r_gap_b)/del_b);
    
    Real term_top = 0.5 * (depth_gap-1) * (sech_term_a - sech_term_b);
    Real term_bot = 0.5 * (depth_gap-1) * tanh_term + 1.0;
    
    return r_polar*term_top/term_bot;
}

Real nu_v(const Real rad, const Real phi, const Real z) {
    Real alpha_R = alphaProfile(rad,phi,z);
    Real r_polar = std::sqrt(rad*rad+z*z);
    Real cs2   = p0_over_r0 * std::pow(r_polar, pslope);
    Real vK    = std::sqrt(gm0/r_polar);
    return alpha_R* cs2 / (vK/r_polar);
}

//----------------------------------------------------------------------------------------
//! background state helpers: Omega's

Real get_Omega_mid(const Real rad, const Real phi, const Real z) {
  Real rho_mid = get_rho_mid(rad,phi,0.0);
  Real dP_dr   = get_dP_mid_dr(rad,phi,0.0);

  Real Omega2 = 1./rad/rho_mid*dP_dr + 1.0/rad/rad/rad;
  return std::sqrt(Omega2);
}

//----------------------------------------------------------------------------------------
//! computes density and velocity in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs2   = get_cs2(rad,phi,0.0);
  Real vK    = std::sqrt(gm0/rad);
  Real alpha_R = alphaProfile(rad,phi,z);
  Real r_polar = std::sqrt(rad*rad+z*z);

  Real nu_v  = alpha_R* cs2 / (vK/rad);
  Real nu_v0 = alphaProfile(r0,0.0,0.0);

  Real Hdisk  = get_H(rad,phi,0.0);
  Real Hdisk0 = get_H(r0,phi,0.0);

  Real Sigma   = Sigma0 * (nu_v0/nu_v) *std::pow(rad, dslope+3.0); 
  Real rho_mid = 0.3989422804 * Sigma / (Hdisk/Hdisk0);
  Real den     = rho_mid * std::exp(gm0/cs2*(std::pow(rad*rad+z*z,-0.5)-1.0/rad));
  
  return std::max(den,dfloor);
}

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs2 = get_cs2(rad,phi,0.0);
  Real vK  = std::sqrt(gm0/rad);
  Real H_over_R_2 = cs2 / (vK*vK);
  Real correction = H_over_R_2 * (3.0 + dlnalpha_dlnR(rad,phi,z));
  return vK * std::sqrt(1.0 - correction);
}

Real vRProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs2 = get_cs2(rad,phi,0.0);
  Real Hdisk = get_H(rad,phi,0.0);
  Real Omega_K = std::sqrt(gm0/rad/rad/rad); 
  Real alpha_R = alphaProfile(rad,phi,z);
  return -alpha_R*cs2/Omega_K/rad * ((dslope*3+(pslope+1.5)*4) + std::pow(z/Hdisk,2)*(2.5*pslope+4.5));
}


}// namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real W, Wphase;
  Real x_a, y_a, z_a, x_ap0, y_ap0, z_ap0, x_ap, y_ap, z_ap;
  Real x_g, y_g, z_g, x_gp0, y_gp0, z_gp0, x_gp, y_gp, z_gp;
  Real rad_ap, phi_ap, rad_gp, phi_gp;
  Real v_r_a, v_theta_a, v_phi_a, v_r_g, v_theta_g, v_phi_g;
  Real v_x_a, v_y_a, v_z_a, v_x_ap, v_y_ap, v_z_ap, v_x_ap0, v_y_ap0, v_z_ap0;
  Real v_x_g, v_y_g, v_z_g, v_rad_gp, v_phi_gp;
  Real v_rad_ap, v_phi_ap, v_x_gp, v_y_gp, v_z_gp, v_x_gp0, v_y_gp0, v_z_gp0;
  Real den, x1, x2, x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
        for (int i=1; i<=ngh; ++i) {
          x1 = pco->x1v(il-i);
                  
          // First check the active cell
                  
          // get the coordinates
          GetCylCoord(pco,rad,phi,z,il,j,k); 
          x_a = rad*cos(phi);
          y_a = rad*sin(phi);
          z_a = z;
          //W = (W_out-W_in)/(R_out-R_in) * (std::sqrt(rad*rad+z*z)-R_in) + W_in;
          W = W_in;
          Wphase = 0.0;
          if (time>time_fix){
            W += dWdt * (time-time_fix);
            Wphase += dpdt * (time-time_fix);
          }
                  
          // apply rotation
          x_ap0 = +x_a*cos(-Wphase) + y_a*sin(-Wphase);
          y_ap0 = -x_a*sin(-Wphase) + y_a*cos(-Wphase);
          z_ap0 = +z_a;
          x_ap = +x_ap0*cos(-W) + z_ap0*sin(-W);
          y_ap = +y_ap0;
          z_ap = -x_ap0*sin(-W) + z_ap0*cos(-W);
          rad_ap = std::sqrt(x_ap*x_ap+y_ap*y_ap);
          phi_ap = atan2(y_ap,x_ap);
                  
          // active cell velocity
          v_r_a     = prim(IM1,k,j,il);
          v_theta_a = prim(IM2,k,j,il);
          v_phi_a   = prim(IM3,k,j,il);
                  
          if (pmb->porb->orbital_advection_defined)
            v_phi_a += vK(pmb->porb, pco->x1v(il), x2, x3);
                  
          // simulation-frame cartesian components of active cell velocity 
          v_x_a = v_r_a*sin(x2)*cos(x3) + v_theta_a*cos(x2)*cos(x3) - v_phi_a*sin(x3);
          v_y_a = v_r_a*sin(x2)*sin(x3) + v_theta_a*cos(x2)*sin(x3) + v_phi_a*cos(x3);
          v_z_a = v_r_a*cos(x2) - v_theta_a*sin(x2);
                  
          // disk-frame cartesian components of active cell velocity 
          v_x_ap0 = +v_x_a*cos(-Wphase) + v_y_a*sin(-Wphase);
          v_y_ap0 = -v_x_a*sin(-Wphase) + v_y_a*cos(-Wphase);
          v_z_ap0 = +v_z_a;
          v_x_ap = +v_x_ap0*cos(-W) + v_z_ap0*sin(-W);
          v_y_ap = +v_y_ap0;
          v_z_ap = -v_x_ap0*sin(-W) + v_z_ap0*cos(-W);
                  
          // disk-frame cylindrical components of active cell velocity 
          v_rad_ap = v_x_ap*cos(phi_ap) + v_y_ap*sin(phi_ap);
          v_phi_ap = -v_x_ap*sin(phi_ap) + v_y_ap*cos(phi_ap);
          v_z_ap   = v_z_ap;
                  
          // apply to the ghost cells
                  
          // get the coordinates
          GetCylCoord(pco,rad,phi,z,il-i,j,k); 
          x_g = rad*cos(phi);
          y_g = rad*sin(phi);
          z_g = z;
                  
          // apply rotation
          x_gp0 = +x_g*cos(-Wphase) + y_g*sin(-Wphase);
          y_gp0 = -x_g*sin(-Wphase) + y_g*cos(-Wphase);
          z_gp0 = +z_g;
          x_gp = +x_gp0*cos(-W) + z_gp0*sin(-W);
          y_gp = +y_gp0;
          z_gp = -x_gp0*sin(-W) + z_gp0*cos(-W);
          rad_gp = std::sqrt(x_gp*x_gp+y_gp*y_gp);
          phi_gp = atan2(y_gp,x_gp);
                  
          // disk-frame cylindrical components of ghost cell velocity 
          v_rad_gp = v_rad_ap * std::pow(rad_ap/rad_gp,0.5);
          v_phi_gp = v_phi_ap * std::pow(rad_ap/rad_gp,0.5);
          v_z_gp   = 0.0; 
                  
          // disk-frame cartesian components of ghost cell velocity 
          v_x_gp = v_rad_gp*cos(phi_gp) - v_phi_gp*sin(phi_gp);
          v_y_gp = v_rad_gp*sin(phi_gp) + v_phi_gp*cos(phi_gp);
          v_z_gp = v_z_gp;
                  
          // simulation-frame cartesian components of ghost cell velocity       
          v_x_gp0 = +v_x_gp*cos(W) + v_z_gp*sin(W);
          v_y_gp0 = v_y_gp;
          v_z_gp0 = -v_x_gp*sin(W) + v_z_gp*cos(W);
          v_x_g = +v_x_gp0*cos(Wphase) + v_y_gp0*sin(Wphase);
          v_y_g = -v_x_gp0*sin(Wphase) + v_y_gp0*cos(Wphase);
          v_z_g = v_z_gp0;  
                  
          // ghost cell velocity in simulation-frame cartesian coordinates
          v_r_g     = v_x_g*sin(x2)*cos(x3) + v_y_g*sin(x2)*sin(x3) + v_z_g*cos(x2);
          v_theta_g = v_x_g*cos(x2)*cos(x3) + v_y_g*cos(x2)*sin(x3) - v_z_g*sin(x2);
          v_phi_g   = -v_x_g*sin(x3) + v_y_g*cos(x3);
                  
          // density
          den = DenProfileCyl(rad_gp,phi_gp,z_gp);
          den = std::max(den,dfloor);
                  
          if (pmb->porb->orbital_advection_defined)
            v_phi_g -= vK(pmb->porb, x1, x2, x3);
                  
          // coordinate conversion
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                            
            // DO NOT USE CYLINDRICAL!! 
                  
            prim(IDN,k,j,il-i) = den;
            prim(IM1,k,j,il-i) = v_r_g*sin(x2) + v_theta_g*cos(x2);
            prim(IM2,k,j,il-i) = v_phi_g;
            prim(IM3,k,j,il-i) = v_r_g*cos(x2) - v_theta_g*sin(x2);
                  
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                            
            prim(IDN,k,j,il-i) = den;
            prim(IM1,k,j,il-i) = v_r_g;
            prim(IM2,k,j,il-i) = v_theta_g;
            prim(IM3,k,j,il-i) = v_phi_g;
                  
          }               
        }
    }
  }
}
                  


//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

// outer: reflect poloidal velocity, maintain rotation
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real W, Wphase;
  Real x_a, y_a, z_a, x_ap0, y_ap0, z_ap0, x_ap, y_ap, z_ap;
  Real x_g, y_g, z_g, x_gp0, y_gp0, z_gp0, x_gp, y_gp, z_gp;
  Real rad_ap, phi_ap, rad_gp, phi_gp;
  Real v_r_a, v_theta_a, v_phi_a, v_r_g, v_theta_g, v_phi_g;
  Real v_x_a, v_y_a, v_z_a, v_x_ap, v_y_ap, v_z_ap, v_x_ap0, v_y_ap0, v_z_ap0;
  Real v_x_g, v_y_g, v_z_g, v_rad_gp, v_phi_gp;
  Real v_rad_ap, v_phi_ap, v_x_gp, v_y_gp, v_z_gp, v_x_gp0, v_y_gp0, v_z_gp0;
  Real den, x1, x2, x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        x1 = pco->x1v(iu+i);

        // First check the active cell

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,iu,j,k); 
        x_a = rad*cos(phi);
        y_a = rad*sin(phi);
        z_a = z;
        //W = (W_out-W_in)/(R_out-R_in) * (std::sqrt(rad*rad+z*z)-R_in) + W_in;
        W = W_out;
        Wphase = 0.0;
        if (time>time_fix){
            W += dWdt * (time-time_fix);
            Wphase += dpdt * (time-time_fix);
        }

        // apply rotation
        x_ap0 = +x_a*cos(-Wphase) + y_a*sin(-Wphase);
        y_ap0 = -x_a*sin(-Wphase) + y_a*cos(-Wphase);
        z_ap0 = +z_a;
        x_ap = +x_ap0*cos(-W) + z_ap0*sin(-W);
        y_ap = +y_ap0;
        z_ap = -x_ap0*sin(-W) + z_ap0*cos(-W);
        rad_ap = std::sqrt(x_ap*x_ap+y_ap*y_ap);
        phi_ap = atan2(y_ap,x_ap);

        // active cell velocity
        v_r_a     = prim(IM1,k,j,iu);
        v_theta_a = prim(IM2,k,j,iu);
        v_phi_a   = prim(IM3,k,j,iu);

        if (pmb->porb->orbital_advection_defined)
          v_phi_a += vK(pmb->porb, pco->x1v(iu), x2, x3);

        // simulation-frame cartesian components of active cell velocity 
        v_x_a = v_r_a*sin(x2)*cos(x3) + v_theta_a*cos(x2)*cos(x3) - v_phi_a*sin(x3);
        v_y_a = v_r_a*sin(x2)*sin(x3) + v_theta_a*cos(x2)*sin(x3) + v_phi_a*cos(x3);
        v_z_a = v_r_a*cos(x2) - v_theta_a*sin(x2);

        // disk-frame cartesian components of active cell velocity 
        v_x_ap0 = +v_x_a*cos(-Wphase) + v_y_a*sin(-Wphase);
        v_y_ap0 = -v_x_a*sin(-Wphase) + v_y_a*cos(-Wphase);
        v_z_ap0 = +v_z_a;
        v_x_ap = +v_x_ap0*cos(-W) + v_z_ap0*sin(-W);
        v_y_ap = +v_y_ap0;
        v_z_ap = -v_x_ap0*sin(-W) + v_z_ap0*cos(-W);


        // disk-frame cylindrical components of active cell velocity 
        v_rad_ap = v_x_ap*cos(phi_ap) + v_y_ap*sin(phi_ap);
        v_phi_ap = -v_x_ap*sin(phi_ap) + v_y_ap*cos(phi_ap);
        v_z_ap   = v_z_ap;

        // apply to the ghost cells

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,iu+i,j,k); 
        x_g = rad*cos(phi);
        y_g = rad*sin(phi);
        z_g = z;

        // apply rotation
        x_gp0 = +x_g*cos(-Wphase) + y_g*sin(-Wphase);
        y_gp0 = -x_g*sin(-Wphase) + y_g*cos(-Wphase);
        z_gp0 = +z_g;
        x_gp = +x_gp0*cos(-W) + z_gp0*sin(-W);
        y_gp = +y_gp0;
        z_gp = -x_gp0*sin(-W) + z_gp0*cos(-W);
        rad_gp = std::sqrt(x_gp*x_gp+y_gp*y_gp);
        phi_gp = atan2(y_gp,x_gp);

        // disk-frame cylindrical components of ghost cell velocity 
        v_rad_gp = v_rad_ap * std::pow(rad_ap/rad_gp,0.5);
        v_phi_gp = v_phi_ap * std::pow(rad_ap/rad_gp,0.5);
        v_z_gp   = 0.0; 

        // disk-frame cartesian components of ghost cell velocity 
        v_x_gp = v_rad_gp*cos(phi_gp) - v_phi_gp*sin(phi_gp);
        v_y_gp = v_rad_gp*sin(phi_gp) + v_phi_gp*cos(phi_gp);
        v_z_gp = v_z_gp;

        // simulation-frame cartesian components of ghost cell velocity       
        v_x_gp0 = +v_x_gp*cos(W) + v_z_gp*sin(W);
        v_y_gp0 = v_y_gp;
        v_z_gp0 = -v_x_gp*sin(W) + v_z_gp*cos(W);
        v_x_g = +v_x_gp0*cos(Wphase) + v_y_gp0*sin(Wphase);
        v_y_g = -v_x_gp0*sin(Wphase) + v_y_gp0*cos(Wphase);
        v_z_g = v_z_gp0;  

        // ghost cell velocity in simulation-frame cartesian coordinates
        v_r_g     = v_x_g*sin(x2)*cos(x3) + v_y_g*sin(x2)*sin(x3) + v_z_g*cos(x2);
        v_theta_g = v_x_g*cos(x2)*cos(x3) + v_y_g*cos(x2)*sin(x3) - v_z_g*sin(x2);
        v_phi_g   = -v_x_g*sin(x3) + v_y_g*cos(x3);

        // density
        den = DenProfileCyl(rad_gp,phi_gp,z_gp);
        den = std::max(den,dfloor);

        if (pmb->porb->orbital_advection_defined)
          v_phi_g -= vK(pmb->porb, x1, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          
          // DO NOT USE CYLINDRICAL!! 

          prim(IDN,k,j,iu+i) = den;
          prim(IM1,k,j,iu+i) = v_r_g*sin(x2) + v_theta_g*cos(x2);
          prim(IM2,k,j,iu+i) = v_phi_g;
          prim(IM3,k,j,iu+i) = v_r_g*cos(x2) - v_theta_g*sin(x2);

        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          
          prim(IDN,k,j,iu+i) = den;
          prim(IM1,k,j,iu+i) = v_r_g;
          prim(IM2,k,j,iu+i) = v_theta_g;
          prim(IM3,k,j,iu+i) = v_phi_g;

        }               
      }
    }
  }
}






//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real rad_W(0.0), phi_W(0.0), z_W(0.0);
  Real x(0.0), y(0.0);
  Real x_W(0.0), y_W(0.0), W;
  Real den_W, vel_W, vel_W_rad, vel_Wx, vel_Wy, vel_Wz;
  Real den, vel_x, vel_y, vel_z, vel_R, vel_phi;
  Real x1, x2, x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=1; j<=ngh; ++j) {
      x2 = pco->x2v(jl-j);
      for (int i=il; i<=iu; ++i) {
        x1 = pco->x1v(i);

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,i,jl-j,k); 
        x = rad*cos(phi);
        y = rad*sin(phi);

        // apply rotation
        W = (W_out-W_in)/(R_out-R_in) * (rad-R_in) + W_in;
        x_W = x;
        y_W = y*cos(-W) - z*sin(-W);
        z_W = y*sin(-W) + z*cos(-W);
        rad_W = std::sqrt(x_W*x_W+y_W*y_W);
        phi_W = atan2(y_W,x_W);

        // compute initial conditions in cylindrical coordinates

        // background
        den_W = DenProfileCyl(rad_W,phi_W,z_W);
        vel_W = VelProfileCyl(rad_W,phi_W,z_W);
        vel_W_rad = vRProfileCyl(rad_W,phi_W,z_W);

        den = den_W; // no need to further rotate scalar
        den = std::max(den,dfloor);

        vel_Wx = cos(phi_W)*vel_W_rad - sin(phi_W)*vel_W;
        vel_Wy = sin(phi_W)*vel_W_rad + cos(phi_W)*vel_W;
        vel_Wz = 0;
        
        vel_x = vel_Wx;
        vel_y = vel_Wy*cos(W) - vel_Wz*sin(W);
        vel_z = vel_Wy*sin(W) + vel_Wz*cos(W);

        vel_R   = (+x*vel_x+y*vel_y)/rad;
        vel_phi = (-y*vel_x+x*vel_y)/rad;        

        if (pmb->porb->orbital_advection_defined)
          vel_phi -= vK(pmb->porb, x1, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          prim(IDN,k,jl-j,i) = den;
          prim(IM1,k,jl-j,i)  = vel_R;
          prim(IM2,k,jl-j,i)  = vel_phi;
          prim(IM3,k,jl-j,i)  = vel_z;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          Real r_polar = std::sqrt(rad*rad+z*z);
          //prim(IDN,k,jl-j,i)  = den;
          //prim(IM1,k,jl-j,i)  = (rad*vel_R + z*vel_z)/r_polar;
          //prim(IM2,k,jl-j,i)  = (z*vel_R - rad*vel_z)/r_polar;
          //prim(IM3,k,jl-j,i)  = vel_phi;
          prim(IDN,k,jl-j,i)  = prim(IDN,k,jl,i);
          prim(IM1,k,jl-j,i)  = prim(IM1,k,jl,i);
          if (prim(IM2,k,jl,i)<0) prim(IM2,k,jl-j,i)=prim(IM2,k,jl,i);
          if (prim(IM2,k,jl,i)>0) prim(IM2,k,jl-j,i)=0.0;
          prim(IM3,k,jl-j,i)  = prim(IM3,k,jl,i);
        }
                 
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real rad_W(0.0), phi_W(0.0), z_W(0.0);
  Real x(0.0), y(0.0);
  Real x_W(0.0), y_W(0.0), W;
  Real den_W, vel_W, vel_W_rad, vel_Wx, vel_Wy, vel_Wz;
  Real den, vel_x, vel_y, vel_z, vel_R, vel_phi;
  Real x1, x2, x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=1; j<=ngh; ++j) {
      x2 = pco->x2v(ju+j);
      for (int i=il; i<=iu; ++i) {
        x1 = pco->x1v(i);

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,i,ju+j,k); 
        x = rad*cos(phi);
        y = rad*sin(phi);

        // apply rotation
        W = W_out;
        x_W = x;
        y_W = y*cos(-W) - z*sin(-W);
        z_W = y*sin(-W) + z*cos(-W);
        rad_W = std::sqrt(x_W*x_W+y_W*y_W);
        phi_W = atan2(y_W,x_W);

        // compute initial conditions in cylindrical coordinates

        // background
        den_W = DenProfileCyl(rad_W,phi_W,z_W);
        vel_W = VelProfileCyl(rad_W,phi_W,z_W);
        vel_W_rad = vRProfileCyl(rad_W,phi_W,z_W);

        den = den_W; // no need to further rotate scalar
        den = std::max(den,dfloor);

        vel_Wx = cos(phi_W)*vel_W_rad - sin(phi_W)*vel_W;
        vel_Wy = sin(phi_W)*vel_W_rad + cos(phi_W)*vel_W;
        vel_Wz = 0;
        
        vel_x = vel_Wx;
        vel_y = vel_Wy*cos(W) - vel_Wz*sin(W);
        vel_z = vel_Wy*sin(W) + vel_Wz*cos(W);

        vel_R   = (+x*vel_x+y*vel_y)/rad;
        vel_phi = (-y*vel_x+x*vel_y)/rad;        

        if (pmb->porb->orbital_advection_defined)
          vel_phi -= vK(pmb->porb, x1, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          prim(IDN,k,ju+j,i) = den;
          prim(IM1,k,ju+j,i)  = vel_R;
          prim(IM2,k,ju+j,i)  = vel_phi;
          prim(IM3,k,ju+j,i)  = vel_z;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          Real r_polar = std::sqrt(rad*rad+z*z);
          //prim(IDN,k,ju+j,i)  = den;
          //prim(IM1,k,ju+j,i)  = (rad*vel_R + z*vel_z)/r_polar;
          //prim(IM2,k,ju+j,i)  = (z*vel_R - rad*vel_z)/r_polar;
          //prim(IM3,k,ju+j,i)  = vel_phi;
          prim(IDN,k,ju+j,i)  = prim(IDN,k,ju,i);
          prim(IM1,k,ju+j,i)  = prim(IM1,k,ju,i);
          if (prim(IM2,k,ju,i)<0) prim(IM2,k,ju+j,i)=prim(IM2,k,ju,i);
          if (prim(IM2,k,ju,i)>0) prim(IM2,k,ju+j,i)=0.0;
          prim(IM3,k,ju+j,i)  = prim(IM3,k,ju,i);          
        }
                 
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = vel;
          prim(IM3,kl-k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = 0.0;
          prim(IM3,kl-k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = vel;
          prim(IM3,ku+k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = 0.0;
          prim(IM3,ku+k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  }
}

int RefinementCondition(MeshBlock *pmb)
{
  AthenaArray<Real> &u = pmb->phydro->u;
  Real rad_max=pmb->pcoord->x1v(pmb->ie);
  Real rad_min=pmb->pcoord->x1v(pmb->is);
  Real tha_max=pmb->pcoord->x2v(pmb->je);
  Real tha_min=pmb->pcoord->x2v(pmb->js);

  Real rho_min = 9999999;
  Real rho_max = 0;
  Real rho_curr = 0;
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
        Real rho_curr = u(IDN,k,j,i);
        rho_min = std::min(rho_min, rho_curr);
        rho_max = std::max(rho_max, rho_curr);

      }
    }
  }

  //printf("%9.12f, %1.9f \n",rho_min, rad_max);

  if ((rad_min>0.8*r_gap_a) || (rad_max<1.2*r_gap_b)){
    if(rho_max > rho_AMR_limit){
      //printf("Refine: %1.9f, %1.9f, (%1.9f, %1.9f), (%1.9f, %1.9f) \n",rho_min, rho_max, rad_min, rad_max, tha_min-1.57, tha_max-1.57);
      //if (rho_min>rho_max) {
      //      printf("-------\n");
      //} 
      return 1;
    } 
    if(rad_max < 0.5*rho_AMR_limit) return -1;
  } else {
    if(rho_max > rho_AMR_limit*depth_gap){
      //printf("Refine: %1.9f, %1.9f, (%1.9f, %1.9f), (%1.9f, %1.9f) \n",rho_min, rho_max, rad_min, rad_max, tha_min-1.57, tha_max-1.57);
      //if (rho_min>rho_max) {
      //      printf("-------\n");
      //} 
      return 1;
    } 
    if(rad_max < 0.5*rho_AMR_limit*depth_gap) return -1;    
  }
  return 0;
}
