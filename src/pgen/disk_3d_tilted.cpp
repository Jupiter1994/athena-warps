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
#include <numeric>    // inner_product
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

// save components of L_in and L_out to history file
Real L_in_component(MeshBlock *pmb, int iout);
Real L_out_component(MeshBlock *pmb, int iout);
// saves debugging variables in history file
//Real Num_inner_cells(MeshBlock *pmb, int iout);
//Real get_M_in(MeshBlock *pmb, int iout); 

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real gapProfile(const Real rad, const Real phi, const Real z); // density gap profile
Real PoverR(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
// custom helper function for the tilted disk
void GetDenVelTilted(Real r, Real theta, Real phi, Real beta, Real &den,
                Real &vr, Real &vtheta, Real &vphi);
// custom helper functions used in Mesh::UserWorkInLoop
void SphToCart(Real r, Real theta, Real phi, Real &x, Real &y, Real &z);
void VelSphToCart(Real theta, Real phi, Real vr, Real vtheta, Real vphi,
	       	Real &vx, Real &vy, Real &vz);
}
namespace {
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas;
Real R_gap, Delta_gap, depth_gap; // gapped density profile parameters
Real dfloor;
Real Omega0;
Real alpha_const; // alpha viscosity parameter
Real W_out; // outer inclination of disk
// variables used for calculating Lhat
Real N_mbs; // (global) number of MeshBlocks in the sim
//Real num_inner_cells; // counts the number of cells at r_in (used for debugging Lhat calculation)
//Real M_in; // mass contained in shell at r_in
Real L_in[3] = {0.0, 0.0, 1.0}; // ang mom vector (Cartesian) at r_in
Real L_out[3] = {0.0, 0.0, 1.0}; // ang mom vector (Cartesian) at r_out
} 

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

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters for density profile
  R_gap     =  pin->GetOrAddReal("problem","R_gap",r0);
  depth_gap = pin->GetOrAddReal("problem","depth_gap",1.0); // default is no gap
  Delta_gap = pin->GetOrAddReal("problem","Delta_gap",0.1);

  // Get parameter for viscosity
  alpha_const = pin->GetReal("problem","alpha_const");

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
 
  // variables for setting boundary conditions
  W_out = pin->GetOrAddReal("problem", "W_out", 0.0);
  L_in[0] = -std::sin(W_out);
  L_in[2] = std::cos(W_out); 
  L_out[0] = -std::sin(W_out);
  L_out[2] = std::cos(W_out);

  // used for debugging Lhat calculation
 // num_inner_cells = 0;
 // M_in = 0;
  // used for history output functions
  N_mbs = 1;

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

  // enroll user-defined viscosity
  EnrollViscosityCoefficient(alpha_viscosity);

  // save L_in and debugging variables to history file
  AllocateUserHistoryOutput(6);
  EnrollUserHistoryOutput(0, L_in_component, "Lx_in");
  EnrollUserHistoryOutput(1, L_in_component, "Ly_in");
  EnrollUserHistoryOutput(2, L_in_component, "Lz_in");
  EnrollUserHistoryOutput(3, L_out_component, "Lx_out");
  EnrollUserHistoryOutput(4, L_out_component, "Ly_out");
  EnrollUserHistoryOutput(5, L_out_component, "Lz_out");
  // debugging output
  //EnrollUserHistoryOutput(3, Num_inner_cells, "Ncells_in");
  //EnrollUserHistoryOutput(4, get_M_in, "M_in");

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), z(0.0); // only used in non-barotropic case
  Real den, vel;
  Real vr(0.0), vtheta(0.0), vphi(0.0); // v's spherical components
  Real r, theta, phi;
  Real gap_R; // density gap profile
  
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    phi = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      theta = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        r = pcoord->x1v(i);
        
	// flat version
	//GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        //gap_R = gapProfile(rad,phi,z);
	//den = DenProfileCyl(rad,phi,z) / gap_R; // apply gap profile
        //vel = VelProfileCyl(rad,phi,z);

	// tilted version
	if (std::cos(phi) > 0) { // x > 0
	    // eg, midplane at theta=80 if W_out=10 deg
	    gap_R = gapProfile(r,phi,r*cos(theta+W_out)); 
	    // debugging vphi calculation for i=0
	    vel = VelProfileCyl(r,phi,r*cos(theta+W_out));
	} else {
	    // eg, midplane at theta=100 if W_out=10
	    gap_R = gapProfile(r,phi,r*cos(theta-W_out));
	    // debugging
	    vel = VelProfileCyl(r,phi,r*cos(theta-W_out));
        }
	// get background values of density and velocity
        GetDenVelTilted(r, theta, phi, W_out, den, vr, vtheta, vphi);
	den /= gap_R; // apply gap profile

	if (porb->orbital_advection_defined)
            vel -= vK(porb, r, theta, phi);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*vr; //0.0;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = den*vel;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = den*vtheta; // 0.0;
	  phydro->u(IM3,k,j,i) = den*vphi; // den*vel;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  return;
}

// at the end of each cycle, calculate this mesh/core's total L_in,
// then calculate+store the global L_in
void Mesh::UserWorkInLoop() {

  // this mesh's contribution to the ang mom (L) vector at r_in
  // (by "mesh", I mean the set of MeshBlocks on a particular
  // process, NOT the whole sim domain)
  // WARNING: L is in Cartesian coordinates.
  Real mesh_L_in[3] = {0.0, 0.0, 0.0};
  // variables that are updated for every MeshBlock (mb) 
  Real den;
  Real r, theta, phi, vr, vtheta, vphi;
  Real dr, dtheta, dphi; // grid spacing
  Real dV; // volume element
  Real x, y, z, vx, vy, vz;
  // debugging variables
  //Real mesh_Ncells_in = 0; // number of cells at r_in (this mesh)
  //Real mesh_M_in = 0; // mass at r_in (this mesh)
  Real mesh_N_mbs = nblocal; // number of MeshBlocks (this mesh)
  
  // this loop calculates this mesh's contribution to L(r_in)
  for (int b=0; b<nblocal; ++b) {
    MeshBlock *pmb = my_blocks(b);
    // primitive variables
    //AthenaArray<Real> &w = pmb->phydro->w;
    // conserved variables
    AthenaArray<Real> &u = pmb->phydro->u;

    // index of r_in in r direction
    int i = pmb->is;

    // debugging
    // printf("b= %d \n", b);
    //printf("mb_in= %.2f \n", pmb->pcoord->x1f(i));

    // ignore MeshBlocks (mbs) that don't include r_in
    if (pmb->loc.lx1 != 0) {
	continue;
    }

    int jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;
    
    // r of the volume-weighted center
    r = pmb->pcoord->x1v(i);  
    // grid spacing (assuming uniform spacing)
    dr = pmb->pcoord->x1v(i+1) - r;
    dtheta = pmb->pcoord->x2v(jl+1) - pmb->pcoord->x2v(jl); 
    dphi = pmb->pcoord->x3v(kl+1) - pmb->pcoord->x3v(kl);  

    // debugging
    //printf("i= %d \n", i);
    //printf("r=%.2f \n", r);

    // adds this mb's L_in to the mesh's L_in
    for (int j=jl; j<=ju; j++) {
      for (int k=kl; k<=ku; k++) {
        // debugging
        //mesh_Ncells_in += 1;
	  
	theta = pmb->pcoord->x2v(j);
	phi = pmb->pcoord->x3v(k);

        den = u(IDN,k,j,i);
	vr = u(IM1,k,j,i) / den;
	vtheta = u(IM2,k,j,i) / den;
	vphi = u(IM3,k,j,i) / den;

        SphToCart(r, theta, phi, x, y, z);
        VelSphToCart(theta, phi, vr, vtheta, vphi,
	     vx, vy, vz);

	// debugging
	//printf("x,y,z=%.2f, %.2f, %.2f, \n", x,y,z);
        //printf("vx,vy,vz=%.2f, %.2f, %.2f, \n", vx,vy,vz);

	// dL = dm(r x v)
	dV = r*r * std::sin(theta) * dr * dtheta * dphi;
	mesh_L_in[0] += den*dV * (y*vz - z*vy);
	mesh_L_in[1] += den*dV * (z*vx - x*vz);
	mesh_L_in[2] += den*dV * (x*vy - y*vx);

	// debugging
	//mesh_M_in += den*dV; // den*dV;
	//printf("j=%d \n", j);
	//printf("theta=%.2f \n", theta);
	//printf("k=%.d \n", k);
	//printf("phi=%.2f \n", phi);
	//printf("dV=%.8f \n", dV);
	//printf("den=%.3f \n", den);
      }
    } // end of j,k loop within this mb

    // debugging
    // printf("mesh_Ncells_in= %.1f \n", mesh_Ncells_in);

  } // end of loop within mesh

  // send L_in to all cores/processes
  #ifdef MPI_PARALLEL
      MPI_Allreduce(&mesh_L_in, &L_in, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(&mesh_Ncells_in, &num_inner_cells, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(&mesh_M_in, &M_in, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      // N_mbs is used in the history output functions
      MPI_Allreduce(&mesh_N_mbs, &N_mbs, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      //std::copy(mesh_L_in, mesh_L_in+3, L_in);
      //num_inner_cells = mesh_Ncells_in;
      //M_in = mesh_M_in;
      
      // debugging
      //printf("L_x= %.4f \n", L_in[0]);
      //printf("L_y= %.4f \n", L_in[1]);
      //printf("L_z= %.4f \n", L_in[2]);
      //printf("M_in= %.3f \n", M_in);
      //printf("N_mbs=%d \n", N_mbs);
      //printf("parallelized, mesh_Ncells_in= %.1f \n", mesh_Ncells_in);
      //printf("num_inner_cells= %.1f \n", num_inner_cells);
  #else // if only using one core
      std::copy(mesh_L_in, mesh_L_in+3, L_in);
      // L_in = mesh_L_in; // sloppy imo
      //num_inner_cells = mesh_Ncells_in;
      printf("1 core, num_inner_cells= %.1f \n", mesh_Ncells_in);
  #endif

  return;
}

//----------------------------------------------------------------------------------------
//! alpha viscosity

void alpha_viscosity(HydroDiffusion *phdif, MeshBlock *pmb,
              const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc,
              int is, int ie, int js, int je,int ks, int ke){
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs2, vK, nu_v;
  Real r; // used in spherical case

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
	  for (int k = pmb->ks; k <= pmb->ke; ++k) {
	  // for (int k = pmb->ks-2; k <= pmb->ke+2; ++k) {
	    for (int j = pmb->js-2; j <= pmb->je+2; ++j) {
	#pragma omp simd
	      for (int i = pmb->is-2; i <= pmb->ie+2; ++i) {
	       // printf("i= %1d \n", i);
	        //printf("j=%1d \n",j);
	        //printf("k=%1d \n",k);
	        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
	        rad   = std::sqrt(rad*rad+z*z);
	        cs2   = p0_over_r0 * std::pow(rad, pslope); // * (1 + 0.5*pslope*std::pow(z/rad,2));
	        vK    = std::sqrt(gm0/rad);
	        //alpha_R = alphaProfile(0.5*(r_gap_a+r_gap_b),phi,z);
		nu_v  = alpha_const* cs2 / (vK/rad);
	        //nu_v  = alpha_0* cs2 / (vK/rad);
	        //printf("%1.9f \n",nu_v);
	        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = nu_v;
	      }
	    }
	  }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
	  for (int i = pmb->is-2; i <= pmb->ie+2; ++i) {
	    r = pmb->pcoord->x1v(i);
	    for (int j = pmb->js-2; j <= pmb->je+2; ++j) {
	#pragma omp simd
	      for (int k = pmb->ks-2; k <= pmb->ke+2; ++k) {
		//GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
	        //rad   = std::sqrt(rad*rad+z*z);
		cs2   = p0_over_r0 * std::pow(r, pslope); // * (1 + 0.5*pslope*std::pow(z/rad,2));
	        vK    = std::sqrt(gm0/r);
	        //alpha_R = alphaProfile(0.5*(r_gap_a+r_gap_b),phi,z);
	        nu_v  = alpha_const* cs2 / (vK/r);
	        //nu_v  = alpha_0* cs2 / (vK/rad);
	        //printf("%1.9f \n",nu_v);
	        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = nu_v;
	      }
	    }
	  }
   }

  return;
}

/*
 * Saves components of L_in to history file; see Mesh::InitUserMeshData.
 */
Real L_in_component(MeshBlock *pmb, int iout) {
  return L_in[iout] / N_mbs;
}

/*
 * Saves components of L_out to history file; should be constant.
 */
Real L_out_component(MeshBlock *pmb, int iout) {
  return L_out[iout-3] / N_mbs;
}

/*
 * Saves the number of cells at r_in to history file. 
 */
//Real Num_inner_cells(MeshBlock *pmb, int iout) {
  //return num_inner_cells / N_mbs;
//}

/*
 * Returns the mass contained within the shell at r_in.
 */
//Real get_M_in(MeshBlock *pmb, int iout) {
  //return M_in / N_mbs;
//}

// namespace for custom helper functions, eg,
// coordinate transformations
namespace {

/*
 * Converts spherical coordinates to Cartesian.
 *
 * r,theta,phi: spherical coordinates
 * x,y,z: references to the (Cartesian) coordinates you're setting
 */
void SphToCart(Real r, Real theta, Real phi, Real &x, Real &y, Real &z) { 
  x = r * std::sin(theta) * std::cos(phi);
  y = r * std::sin(theta) * std::sin(phi);
  z = r * std::cos(theta);
  
  return;

}

/**
 * Converts Cartesian coordinates to spherical.
 *
 * x,y,z: Cartesian coordinates
 * r,theta,phi: references to the spherical coordinates you're setting
 */
void CartToSph(Real x, Real y, Real z, Real &r, Real &theta, Real &phi) {

  r = std::sqrt(x*x + y*y + z*z);
  theta = std::atan2(std::hypot(x,y), z); // range: [0, pi]
  phi = std::atan2(y, x); // range: [-pi, pi]

  return;

}

/**
* Converts a spherical velocity vector to Cartesian coordinates. Note that 
* this requires the position vector's theta and phi coordinates.
*/
void VelSphToCart(Real theta, Real phi, Real vr, Real vtheta, Real vphi,
	       	Real &vx, Real &vy, Real &vz) {

  // spherical unit vectors  
  std::vector<Real> rHat = 
      {std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi), std::cos(theta)};
  std::vector<Real> thetaHat = 
      {std::cos(theta)*std::cos(phi), std::cos(theta)*std::sin(phi), -std::sin(theta)};
  std::vector<Real> phiHat = {-std::sin(phi), std::cos(phi), 0};

  // Cartesian velocity vector (v = sum over v_i * ihat)
  vx = vr*rHat[0] + vtheta*thetaHat[0] + vphi*phiHat[0];
  vy = vr*rHat[1] + vtheta*thetaHat[1] + vphi*phiHat[1];
  vz = vr*rHat[2] + vtheta*thetaHat[2] + vphi*phiHat[2];

  return;

}

/**
* Converts a Cartesian velocity vector to spherical coordinates. Note that
* this requires the position vector's theta and phi coordinates.
*/
void VelCartToSph(Real theta, Real phi, Real vx, Real vy, Real vz,
	       	Real &vr, Real &vtheta, Real &vphi) {

  // Cartesian velocity vector
  std::vector<Real> vCart = {vx, vy, vz};

  // spherical unit vectors
  std::vector<Real> rHat =
      {std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi), std::cos(theta)};
  std::vector<Real> thetaHat =
      {std::cos(theta)*std::cos(phi), std::cos(theta)*std::sin(phi), -std::sin(theta)};
  std::vector<Real> phiHat = {-std::sin(phi), std::cos(phi), 0};

  //vr = std::inner_product(vCart.begin(), vCart.end(), rHat.begin(), 0);
  //vtheta = std::inner_product(vCart.begin(), vCart.end(), thetaHat.begin(), 0);
  //vphi = std::inner_product(vCart.begin(), vCart.end(), phiHat.begin(), 0);

  // eg, vr = dot product of r-hat with position vector
  vr = vx*rHat[0] + vy*rHat[1] + vz*rHat[2];
  vtheta = vx*thetaHat[0] + vy*thetaHat[1] + vz*thetaHat[2];
  vphi = vx*phiHat[0] + vy*phiHat[1] + vz*phiHat[2];

  // debugging
  //printf("vphi from VCTS: %.2f \n",vphi);

  return;

}

/**
 * Converts spherical coordinates to cylindrical.
 * (Unlike GetCylCoord(), this function allows you to
 * input an arbitrary set of spherical coordinates.)
 *
 * r,theta,phi: spherical coordinates
 * R,Phi,z: references to the (cylindrical) coordinates you're setting
 */
void SphToCyl(Real r, Real theta, Real phi, Real &R, Real &Phi, Real &z) {
  R = r * std::sin(theta);
  Phi = phi;
  z = r * std::cos(theta);

  return;

}

/**
 * Rotate a vector (in Cartesian coordinates) by the angle `beta` about the y-axis.
 * (We're rotating the xz-plane counterclockwise; e.g., the point (1,0) becomes
 * (0,1) after a 90-degree rotation.)
 *
 * x,y,z: references to the Cartesian coordinates you're changing
 * beta: the angle by which you're rotating the vector
 */
void RotateAroundY(Real &x, Real &y, Real &z, Real beta) {

  Real x_new, z_new; // x,z coords of rotated vector

  // apply rotation around y-axis
  x_new = std::cos(beta)*x - std::sin(beta)*z;
  z_new = std::sin(beta)*x + std::cos(beta)*z;

  x = x_new;
  z = z_new;

  return;
}

/**
 * Given a set of spherical coordinates and a rotation angle `beta`,
 * return the background values of density and velocity at those coordinates.
 * (Note: Does NOT include accretion. v_r is set to 0.)
 *
 * r,theta,phi: spherical coordinates
 * beta: rotation angle
 * rho,vr,vtheta,vphi: references to the background values you're setting
 */
void GetDenVelTilted(Real r, Real theta, Real phi, Real beta, Real &den,
		Real &vr, Real &vtheta, Real &vphi) {
  //Real rad(0.0), Phi(0.0), z(0.0);
  Real x, y, z;
  Real r_, theta_, phi_; // spherical coords of rotated position vector
  Real rad_, Phi_, z_; // cylindrical coords in rotated frame
  Real vel;
  Real vx, vy, vz;

  // calculate values at an angle beta "below" your position
  // first, get spherical coordinates at that new position
  SphToCart(r, theta, phi, x, y, z);
  RotateAroundY(x, y, z, -beta);
  CartToSph(x, y, z, r_, theta_, phi_);
  // second, compute steady-state den+vel in the rotated frame
  SphToCyl(r_, theta_, phi_, rad_, Phi_, z_);
  den = DenProfileCyl(rad_, Phi_, z_);
  vel = VelProfileCyl(rad_, Phi_, z_);
  // assume R~r
  //den = DenProfileCyl(r_,phi_,z);
  //vel = VelProfileCyl(r_,phi_,z);

  // compute the velocity vector in rotated frame
  vx = -vel * std::sin(phi_);
  vy = vel * std::cos(phi_);
  vz = 0;
  // get velocity at (r,theta,phi) in the de-rotated frame
  RotateAroundY(vx, vy, vz, beta);
  VelCartToSph(theta, phi, vx, vy, vz, vr, vtheta, vphi);

  // debugging
  //printf("r,theta,phi=%.2f, %.2f, %.2f, \n", r,theta,phi);
  //printf("vel,vphi=%.2f, %.2f, \n", vel,vphi);
  //printf("vr,vtheta=%.2f, %.2f, \n", vr,vtheta);
  //printf("vx,vy=%.2f, %.2f, \n", vx,vy);

  return;
}

/** 
 * Given a point in spherical coordinates (r,theta,phi) and the local angular 
 * momentum vector (L), calculate the point's distance from the midplane (z_mp).
 */
void GetZfromL(Real r, Real theta, Real phi, Real L[], Real &z_mp) {

  Real x, y, z;
  Real Lmag;

  // magnitude of L
  Lmag = std::sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2]);

  // get Cartesian coordinates
  SphToCart(r, theta, phi, x, y, z);

  // z_mp is the dot product of Lhat with the position vector
  z_mp = L[0]*x + L[1]*y + L[2]*z;
  z_mp /= Lmag;

  // debugging: for W_out=0, this is the correct z_mp 
  // z_mp = z;

  return;
}

//----------------------------------------------------------------------------------------
//! computes radial velocity (v_r) for a 3D disk in steady-state, given the disk-frame
// coordinates (rad, phi, z)

Real VrProfileCyl(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverR(rad, phi, z);
  Real vK = std::sqrt(gm0/rad);
  Real z_over_H = z / std::sqrt(p_over_r) * (vK/rad); // H=cs/Omega_K, approximately
  Real vr = -alpha_const*p_over_r/vK * (-3 + 4.5*SQR(z_over_H));  
  return vr;
}

} // namespace for custom helper functions


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
//! background state helpers: gap shape
//
Real gapProfile(const Real rad, const Real phi, const Real z) {
    Real r_polar = rad; //  std::sqrt(rad*rad+z*z);
    Real exponent = -std::pow((r_polar - R_gap), 2) / (2 * std::pow(Delta_gap, 2));
    return 1 + (depth_gap - 1) * std::exp(exponent);
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  //Real gap_R = gapProfile(rad,phi,z);

  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem; // * (1.0/gap_R);

  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverR(rad, phi, z);
  // warning: if dslope isn't -1.5, the below line won't be correct for
  // steady-state
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}
} // namespace (for provided Athena++ functions)

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0); // cylindrical coordinates at active cell (in disk's frame)
  Real vel;
  Real rad_gh, z_gh; // cylindrical radius and height at ghost cell
  Real r_ac, r_gh; // radial coords of active and ghost cells
  Real z_ac; // cylindrical height at active cell
  Real theta; // polar coordinate
  Real den_gh, vr, vtheta, vphi; // background den, vel at ghost cell (spherical case)
  Real W_in; // tilt at R_in

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  // printf("ngh= %1d \n", ngh);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad_gh,phi,z,il-i,j,k);
	  GetCylCoord(pco,rad,phi,z,il,j,k);
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il) * std::pow(rad_gh/rad,-1.5);
          vel = VelProfileCyl(rad,phi,z); // not used
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = prim(IM1,k,j,il) * std::pow(rad_gh/rad,0.5);
	  // below line excludes pressure correction to v_phi
	  prim(IM2,k,j,il-i) = prim(IM2,k,j,il) * std::pow(rad_gh/rad,-0.5);
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
	r_ac = pco->x1v(il);
	theta = pco->x2v(j);
	phi = pco->x3v(k);
	GetZfromL(r_ac, theta, phi, L_in, z_ac);
	//rad = std::sqrt(r*r - z*z); 
	for (int i=1; i<=ngh; ++i) {
	  r_gh = pco->x1v(il-i);

          //GetCylCoord(pco,rad_gh,phi,z_gh,il-i,j,k);
          //GetCylCoord(pco,rad,phi,z,il,j,k); 
	
	  GetZfromL(r_gh, theta, phi, L_in, z_gh);
          //rad_gh = std::sqrt(r_gh*r_gh - z_gh*z_gh);
	  W_in = std::asin(-L_in[0] / L_in[2]); // arcsin(-L_x/L_z)
	  GetDenVelTilted(r_gh, theta, phi, W_in, den_gh, vr, vtheta, vphi);

          prim(IDN,k,j,il-i) = den_gh; // prim(IDN,k,j,il) *
	     //DenProfileCyl(r_gh,phi,z_gh)/DenProfileCyl(r_ac,phi,z_ac); // assume r~R
	  // vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = prim(IM1,k,j,il) * std::pow(r_gh/r_ac,0.5); // VrProfileCyl(r_gh,phi,z_gh);
		// VrProfileCyl(r_gh,phi,z_gh)/VrProfileCyl(r_ac,phi,z_ac); // assume r~R
	  //prim(IM1,k,j,il-i) = prim(IM1,k,j,il) * std::pow(rad_gh/rad,0.5); // v_r
          prim(IM2,k,j,il-i) = vtheta; // prim(IM2,k,j,il) * std::pow(r_gh/r_ac,-0.5); // 0.0;  // v_theta
          prim(IM3,k,j,il-i) = vphi; // prim(IM3,k,j,il) * std::pow(r_gh/r_ac,-0.5); // v_phi; assume r~R
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0); // cyl. coords at active cell (in disk frame)
  Real rad_gh, z_gh; // cylindrical R and z at ghost cell (in disk frame)
  Real r, r_gh; // spherical radii of last active and ghost cells, respectively
  Real theta; // polar angle in spherical coordinates 
  Real den, vel;
  Real den_gh, vr, vtheta, vphi; // used in spherical case
  Real vK_gh; // Keplerian velocity in ghost cell
  Real z_over_H; // z/H (used if coord sys is spherical)
  Real r_ac;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad_gh,phi,z,iu+i,j,k);
	  GetCylCoord(pco,rad,phi,z,iu,j,k);

	  den = DenProfileCyl(rad_gh,phi,z); // slightly incorrect if dslope != -1.5
	  den = std::max(den,dfloor);
	  prim(IDN,k,j,iu+i) = den; // hold fixed at steady-state Sigma value
          //prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu) * std::pow(rad_gh/rad,-1.5);

          vel = VelProfileCyl(rad,phi,z); // ignore since no orbital advection
	  if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          //prim(IM1,k,j,iu+i) = prim(IM1,k,j,iu) * std::pow(rad_gh/rad,0.5);
	  vK_gh = std::sqrt(gm0/rad_gh);
	  prim(IM1,k,j,iu+i) = -1.5*alpha_const*p0_over_r0/vK_gh; // hold fixed
	  prim(IM2,k,j,iu+i) = VelProfileCyl(rad_gh,phi,z); // slightly incorrect if dslope!=-1.5
	  prim(IM3,k,j,iu+i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      phi = pco->x3v(k);
      for (int j=jl; j<=ju; ++j) {
        theta = pco->x2v(j);
	for (int i=1; i<=ngh; ++i) {
          //GetCylCoord(pco,rad_gh,phi,z_gh,iu+i,j,k);
          r_ac = pco->x1v(iu); 
          //rad_gh = std::sqrt(r_gh*r_gh - z_gh*z_gh);
          r_gh = pco->x1v(iu+i);
          GetZfromL(r_gh, theta, phi, L_out, z_gh);
	  //rad_gh = std::sqrt(r_gh*r_gh - z_gh*z_gh);

          GetDenVelTilted(r_gh, theta, phi, W_out, den_gh, vr, vtheta, vphi);

	  prim(IDN,k,j,iu+i) = den_gh; // DenProfileCyl(rad_gh,phi,z_gh); //  hold the outer rho fixed
	  // vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
	  // v_r determined by steady-state accretion, not GetDenVelTilted
          prim(IM1,k,j,iu+i) = prim(IM1,k,j,iu) * std::pow(r_gh/r_ac,0.5); // VrProfileCyl(r_gh,phi,z_gh); // v_r; assume R~r
          //vK_gh = std::sqrt(gm0/rad_gh);
	  //z_over_H = z_gh / std::sqrt(p0_over_r0) * (vK_gh/rad_gh); // H=cs/Omega
          //prim(IM1,k,j,iu+i) = -alpha_const*p0_over_r0/vK_gh * (-3 + 4.5*SQR(z_over_H)); // v_r
	  prim(IM2,k,j,iu+i) = vtheta; // 0.0; // v_theta
          prim(IM3,k,j,iu+i) = vphi; // VelProfileCyl(rad_gh,phi,z_gh); // v_phi
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
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
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = vel;
          prim(IM3,k,jl-j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = 0.0;
          prim(IM3,k,jl-j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
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
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = vel;
          prim(IM3,k,ju+j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = 0.0;
          prim(IM3,k,ju+j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
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
