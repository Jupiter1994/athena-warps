//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk_2d_viscous_gap.cpp
//! \brief Initializes Keplerian accretion disk in 2D cylindrical coordinations. A density 
//! gap is added in the middle part of the sim. Must be globally isothermal now.
//! 
//! For the main updates, see:
//! Lines xxx-xxx
//!

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

// -- Functions and parameters needed for this setup.
// 1) basic orbital dynamics
Real gm0, r0, Omega0;
// 2) for the (gapped) density profile
Real Sigma0, dslope, dfloor;
Real r_1, r_2, xi_1, xi_2;
Real R_in, R_out, W_in, W_out, dWdt, dpdt, time_fix;
Real R_gap, Delta_gap, depth_gap;
// 3) the sound speed profile (must be globally isothermal now)
Real p0_over_r0, pslope, gamma_gas;
// 4) viscosity
Real alpha_vis;
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

  R_gap     =  pin->GetOrAddReal("problem","R_gap",1.5);
  depth_gap = pin->GetOrAddReal("problem","depth_gap",2.0);
  Delta_gap = pin->GetOrAddReal("problem","Delta_gap",0.1);

  // Get parameters for alpha viscosity
  alpha_vis   = pin->GetOrAddReal("problem","alpha_vis",0.0);

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
  if (alpha_vis>1e-8){
    EnrollViscosityCoefficient(alpha_viscosity); // alpha viscosity
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel_R, vel_phi, vel_z;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);

        // --- get the  coordinates
        GetCylCoord(pcoord,rad,phi,z,i,j,k); 

        // --- compute initial conditions in cylindrical coordinates
        den     = DenProfileCyl(rad,phi,z);
        den     = std::max(den,dfloor);
        vel_phi = VelProfileCyl(rad,phi,z);
        vel_R   = 0.0; 
        vel_z   = 0.0; 

        if (porb->orbital_advection_defined)
          vel_phi -= vK(porb, x1, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IDN,k,j,i) = den;
          phydro->u(IM1,k,j,i) = den*vel_R;
          phydro->u(IM2,k,j,i) = den*vel_phi;
          phydro->u(IM3,k,j,i) = den*vel_z;
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
  Real cs2, vK, nu_v;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
  //for (int k = pmb->ks-2; k <= pmb->ke+2; ++k) {
    for (int j = pmb->js-2; j <= pmb->je+2; ++j) {
#pragma omp simd
      for (int i = pmb->is-2; i <= pmb->ie+2; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); 
        vK    = std::sqrt(gm0/rad);
        cs2   = p0_over_r0 * std::pow(rad, pslope); 
        nu_v  = alpha_vis* cs2 / (vK/rad);
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = nu_v;
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! Locally isothermal cooling -- WIP

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
}

Real get_dcs2_dr(const Real rad, const Real phi, const Real z) {
  return p0_over_r0 * pslope * std::pow(rad, pslope-1) * (1 + 0.5*pslope*std::pow(z/rad,2));
}

Real get_d2cs2_dr2(const Real rad, const Real phi, const Real z) {
  return p0_over_r0 * pslope * (pslope-1) * std::pow(rad, pslope-2) * (1 + 0.5*pslope*std::pow(z/rad,2));
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
//! background state helpers: gap shape
//
Real gapProfile(const Real rad, const Real phi, const Real z) {
    Real r_polar = rad; //  std::sqrt(rad*rad+z*z);
    Real exponent = -std::pow((r_polar - R_gap), 2) / (2 * std::pow(Delta_gap, 2));
    return 1 + (depth_gap - 1) * std::exp(exponent);
}

Real dlngap_dlnR(const Real rad, const Real phi, const Real z) {
    Real gap_R = gapProfile(rad,phi,z);
    return (1 / gap_R - 1.0) * rad * (rad - R_gap) / std::pow(Delta_gap, 2);
}

Real nu_v(const Real rad, const Real phi, const Real z) {
    Real gap_R = gapProfile(rad,phi,z);
    Real r_polar = rad; //std::sqrt(rad*rad+z*z);
    Real cs2   = p0_over_r0 * std::pow(r_polar, pslope);
    Real vK    = std::sqrt(gm0/r_polar);
    return gap_R* cs2 / (vK/r_polar);
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
  Real gap_R = gapProfile(rad,phi,z);

  Real Hdisk  = get_H(rad,phi,0.0);
  Real Hdisk0 = get_H(r0,phi,0.0);

  Real Sigma   = Sigma0 * (1.0/gap_R) *std::pow(rad, dslope); 
  Real den = Sigma;

  return std::max(den,dfloor);
}

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs2 = get_cs2(rad,phi,0.0);
  Real vK  = std::sqrt(gm0/rad);
  Real H_over_R_2 = cs2 / (vK*vK);
  Real correction = H_over_R_2 * (1.5 + dlngap_dlnR(rad,phi,z));
  return vK * std::sqrt(1.0 - correction);
}

Real vRProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs2 = get_cs2(rad,phi,0.0);
  Real Hdisk = get_H(rad,phi,0.0);
  Real Omega_K = std::sqrt(gm0/rad/rad/rad); 
  return -alpha_vis*cs2/Omega_K/rad * ((dslope*3+(pslope+1.5)*4) + std::pow(z/Hdisk,2)*(2.5*pslope+4.5));
}


}// namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real W, cost, sint, n_x, n_y, n_z;
  Real den, v_r_g, v_theta_g, v_phi_g;
  Real x1, x1a, x1g, x2, x3;
  Real vr_a2g, vphi_a2g, rho_a2g, vKa, vKg;
  Real x_a, y_a, z_a, dist_a, expo_a;
  Real x_g, y_g, z_g, dist_g, expo_g;
  Real Sigma_a, Sigma_g, dx1,dx2,dx3, Delta_x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        x1g = pco->x1v(il-i);
        x1a = pco->x1v(il);
        vKa = std::sqrt(gm0/x1a);
        vKg = std::sqrt(gm0/x1g);

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,il-i,j,k);   

        // set the velocities
        v_r_g = prim(IM1,k,j,il) * std::pow(x1g/x1a,0.5);
        v_phi_g = prim(IM2,k,j,il)/VelProfileCyl(x1a,phi,z) * VelProfileCyl(x1g,phi,z);
    
        // get density from Sigma extrapolation
        den = prim(IDN,k,j,il) * std::pow(x1g/x1a,-1.5) ;
        den = std::max(den,dfloor);

        if (pmb->porb->orbital_advection_defined)
          v_phi_g -= vK(pmb->porb, x1g, x2, x3);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {

          prim(IDN,k,j,il-i) = den;
          prim(IM1,k,j,il-i) = v_r_g;
          prim(IM2,k,j,il-i) = v_phi_g;
          prim(IM3,k,j,il-i) = 0.0;

        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          
          // DO NOT USE SPHERICAL...

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
  Real W, cost, sint, n_x, n_y, n_z;
  Real den, v_r_g, v_theta_g, v_phi_g;
  Real x1, x1a, x1g, x2, x3;
  Real vr_a2g, vphi_a2g, rho_a2g, vKa, vKg;
  Real x_a, y_a, z_a, dist_a, expo_a;
  Real x_g, y_g, z_g, dist_g, expo_g;
  Real Sigma_a, Sigma_g, dx1,dx2,dx3, Delta_x3;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        x1g = pco->x1v(iu+i);
        x1a = pco->x1v(iu);
        vKa = std::sqrt(gm0/x1a);
        vKg = std::sqrt(gm0/x1g);

        // get the coordinates
        GetCylCoord(pco,rad,phi,z,iu+i,j,k);   

        // set the velocities
        v_r_g = -alpha_vis*p0_over_r0/vKg * 1.5;
        v_phi_g = VelProfileCyl(rad,phi,z);
    
        // get density from Sigma extrapolation
        den = DenProfileCyl(rad,phi,z);
        den = std::max(den,dfloor);

        // coordinate conversion
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          
          prim(IDN,k,j,iu+i) = den;
          prim(IM1,k,j,iu+i) = v_r_g;
          prim(IM2,k,j,iu+i) = v_phi_g;
          prim(IM3,k,j,iu+i) = 0.0;

        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {

          // DO NOT USE SPHERICAL...

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

