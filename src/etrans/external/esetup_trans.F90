SUBROUTINE ESETUP_TRANS(KMSMAX,KSMAX,KDGL,KDGUX,KLOEN,LDSPLIT,&
 & KFLEV,KTMAX,KRESOL,PEXWN,PEYWN,PWEIGHT,LDGRIDONLY,KNOEXTZL,KNOEXTZG, &
 & LDUSEFFTW)
!**** *ESETUP_TRANS* - Setup transform package for specific resolution

!     Purpose.
!     --------
!     To setup for making spectral transforms. Each call to this routine
!     creates a new resolution up to a maximum of NMAX_RESOL set up in
!     SETUP_TRANS0. You need to call SETUP_TRANS0 before this routine can
!     be called.

!**   Interface.
!     ----------
!     CALL ESETUP_TRANS(...)

!     Explicit arguments : KLOEN,LDSPLIT are optional arguments
!     -------------------- 
!     KSMAX - spectral truncation required
!     KDGL  - number of Gaussian latitudes
!     KLOEN(:) - number of points on each Gaussian latitude [2*KDGL]
!     LDSPLIT - true if split latitudes in grid-point space [false]
!     KTMAX - truncation order for tendencies?
!     KRESOL - the resolution identifier
!     KSMAX,KDGL,KTMAX and KLOEN are GLOBAL variables desribing the resolution
!     in spectral and grid-point space
!     LDGRIDONLY - true if only grid space is required


!     LDSPLIT describe the distribution among processors of
!     grid-point data and has no relevance if you are using a single processor
 
!     LDUSEFFTW   - Use FFTW for FFTs

!     Method.
!     -------

!     Externals.  ESET_RESOL   - set resolution
!     ----------  ESETUP_DIMS  - setup distribution independent dimensions
!                 SUEMP_TRANS_PRELEG - first part of setup of distr. environment
!                 SULEG - Compute Legandre polonomial and Gaussian
!                         Latitudes and Weights
!                 ESETUP_GEOM - Compute arrays related to grid-point geometry
!                 SUEMP_TRANS - Second part of setup of distributed environment
!                 SUEFFT - setup for FFT

!     Author.
!     -------
!        Mats Hamrud *ECMWF*

!     Modifications.
!     --------------
!        Original : 00-03-03
!        02-04-11 A. Bogatchev: Passing of TCDIS
!        02-11-14 C. Fischer: soften test on KDGL
!        M.Hamrud      01-Oct-2003 CY28 Cleaning
!        A.Nmiri       15-Nov-2007 Phasing with TFL 32R3
!        A.Bogatchev   16-Sep-2010 Phasing cy37
!        D. Degrauwe  (Feb 2012): Alternative extension zone (E')
!        R. El Khatib 02-Mar-2012 Support for mixed multi-resolutions
!        R. El Khatib 09-Aug-2012 %LAM in GEOM_TYPE
!        R. El Khatib 14-Jun-2013 LENABLED
!        R. El Khatib 01-Sep-2015 Support for FFTW
!     ------------------------------------------------------------------

USE PARKIND1  ,ONLY : JPIM     ,JPRB
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK, JPHOOK

!ifndef INTERFACE

USE TPM_GEN         ,ONLY : NERR, NOUT, NPRINTLEV, MSETUP0,    &
     &                      NCUR_RESOL, NDEF_RESOL, NMAX_RESOL, LENABLED
USE TPM_DIM         ,ONLY : R, DIM_RESOL, R_NSMAX,R_NTMAX, R_NDGNH, R_NDGL, R_NNOEXTZL
USE TPM_DISTR       ,ONLY : D, DISTR_RESOL,NPROC,D_NUMP,D_MYMS,D_NSTAGT0B,D_NSTAGT1B,&
     &                      D_NPROCL,D_NPNTGTB1,D_NASM0,D_NSTAGTF,D_MSTABF,D_NPNTGTB0,&
     &                      D_NPROCM,D_NPTRLS
USE TPM_GEOMETRY    ,ONLY : G, GEOM_RESOL, G_NDGLU, G_NMEN, G_NMEN_MAX,G_NLOEN, G_NLOEN_MAX
USE TPM_FIELDS      ,ONLY : FIELDS_RESOL, F
USE TPM_FFT         ,ONLY : T, FFT_RESOL !, TB, FFTB_RESOL
#ifdef WITH_FFTW
USE TPM_FFTW        ,ONLY : TW, FFTW_RESOL
#endif
USE TPM_FFTC        ,ONLY : TC, FFTC_RESOL
USE TPM_FLT         ,ONLY : FLT_RESOL

USE TPMALD_DIM      ,ONLY : RALD, ALDDIM_RESOL
USE TPMALD_DISTR    ,ONLY : ALDDISTR_RESOL
USE TPMALD_FFT      ,ONLY : TALD, ALDFFT_RESOL
USE TPMALD_FIELDS   ,ONLY : ALDFIELDS_RESOL
USE TPMALD_GEO      ,ONLY : GALD, ALDGEO_RESOL

USE ESET_RESOL_MOD  ,ONLY : ESET_RESOL
USE ESETUP_DIMS_MOD ,ONLY : ESETUP_DIMS
USE SUEMP_TRANS_MOD ,ONLY : SUEMP_TRANS
USE SUEMP_TRANS_PRELEG_MOD ,ONLY : SUEMP_TRANS_PRELEG
!USE SULEG_MOD
USE ESETUP_GEOM_MOD ,ONLY : ESETUP_GEOM
USE SUEFFT_MOD      ,ONLY : SUEFFT
USE ABORT_TRANS_MOD ,ONLY : ABORT_TRANS
#ifdef _OPENACC
use openacc
#endif

!endif INTERFACE

IMPLICIT NONE

! Dummy arguments
INTEGER(KIND=JPIM),INTENT(IN)    :: KMSMAX
INTEGER(KIND=JPIM),INTENT(IN)    :: KSMAX
INTEGER(KIND=JPIM),INTENT(IN)    :: KDGL
INTEGER(KIND=JPIM),INTENT(IN)    :: KDGUX
INTEGER(KIND=JPIM),INTENT(IN)    :: KLOEN(:)
LOGICAL           ,OPTIONAL,INTENT(IN)    :: LDSPLIT
LOGICAL           ,OPTIONAL,INTENT(IN)    :: LDGRIDONLY
INTEGER(KIND=JPIM),OPTIONAL,INTENT(IN)    :: KTMAX
INTEGER(KIND=JPIM),OPTIONAL,INTENT(IN)    :: KFLEV
INTEGER(KIND=JPIM),OPTIONAL,INTENT(INOUT) :: KRESOL
REAL(KIND=JPRB)   ,OPTIONAL,INTENT(IN)    :: PEXWN
REAL(KIND=JPRB)   ,OPTIONAL,INTENT(IN)    :: PEYWN
REAL(KIND=JPRB)   ,OPTIONAL,INTENT(IN)    :: PWEIGHT(:)
INTEGER(KIND=JPIM),OPTIONAL,INTENT(IN)    :: KNOEXTZL
INTEGER(KIND=JPIM),OPTIONAL,INTENT(IN)    :: KNOEXTZG
LOGICAL   ,OPTIONAL,INTENT(IN)            :: LDUSEFFTW

!ifndef INTERFACE

! Local variables
LOGICAL :: LLP1,LLP2
INTEGER(KIND=JPIM) :: I, J
REAL(KIND=JPHOOK) :: ZHOOK_HANDLE

!     ------------------------------------------------------------------

IF (LHOOK) CALL DR_HOOK('ESETUP_TRANS',0,ZHOOK_HANDLE)

IF(MSETUP0 == 0) THEN
  CALL ABORT_TRANS('ESETUP_TRANS: SETUP_TRANS0 HAS TO BE CALLED BEFORE ESETUP_TRANS')
ENDIF
LLP1 = NPRINTLEV>0
LLP2 = NPRINTLEV>1
IF(LLP1) WRITE(NOUT,*) '=== ENTER ROUTINE ESETUP_TRANS ==='

! Allocate resolution dependent structures common to global and LAM
IF(.NOT. ALLOCATED(DIM_RESOL)) THEN
  NDEF_RESOL = 1
  ALLOCATE(DIM_RESOL(NMAX_RESOL))
  ALLOCATE(FIELDS_RESOL(NMAX_RESOL))
  ALLOCATE(GEOM_RESOL(NMAX_RESOL))
  ALLOCATE(DISTR_RESOL(NMAX_RESOL))
  ALLOCATE(FFT_RESOL(NMAX_RESOL))
  !ALLOCATE(FFTB_RESOL(NMAX_RESOL))
#ifdef WITH_FFTW
  ALLOCATE(FFTW_RESOL(NMAX_RESOL))
#endif
  ALLOCATE(FFTC_RESOL(NMAX_RESOL))
  ALLOCATE(FLT_RESOL(NMAX_RESOL))
  GEOM_RESOL(:)%LAM=.FALSE.
  ALLOCATE(LENABLED(NMAX_RESOL))
  LENABLED(:)=.FALSE.
ELSE
  NDEF_RESOL = NDEF_RESOL+1
  IF(NDEF_RESOL > NMAX_RESOL) THEN
    CALL ABORT_TRANS('ESETUP_TRANS:NDEF_RESOL > NMAX_RESOL')
  ENDIF
ENDIF
! Allocate LAM-specific resolution dependent structures
IF(.NOT. ALLOCATED(ALDDIM_RESOL)) THEN
  ALLOCATE(ALDDIM_RESOL(NMAX_RESOL))
  ALLOCATE(ALDFIELDS_RESOL(NMAX_RESOL))
  ALLOCATE(ALDGEO_RESOL(NMAX_RESOL))
  ALLOCATE(ALDDISTR_RESOL(NMAX_RESOL))
  ALLOCATE(ALDFFT_RESOL(NMAX_RESOL))
ENDIF


IF (PRESENT(KRESOL)) THEN
  KRESOL=NDEF_RESOL
ENDIF

! Point at structures due to be initialized
CALL ESET_RESOL(NDEF_RESOL)
IF(LLP1) WRITE(NOUT,*) '=== DEFINING RESOLUTION ',NCUR_RESOL

! Defaults for optional arguments

G%LREDUCED_GRID = .FALSE.
D%LGRIDONLY = .FALSE.
D%LSPLIT = .FALSE.
TALD%LFFT992=.TRUE. ! Use FFT992 interface for FFTs
#ifdef WITH_FFTW
TW%LFFTW=.FALSE. ! Use FFTW interface for FFTs
#endif

! NON-OPTIONAL ARGUMENTS
R%NSMAX = KSMAX
RALD%NMSMAX=KMSMAX
RALD%NDGUX=KDGUX
R%NDGL  = KDGL
RALD%NDGLSUR=KDGL+2
R%NDLON =KLOEN(1)

! IMPLICIT argument :
G%LAM = .TRUE.

IF (KDGL <= 0) THEN
  CALL ABORT_TRANS ('ESETUP_TRANS: KDGL IS NOT A POSITIVE NUMBER')
ENDIF

! Optional arguments

ALLOCATE(G%NLOEN(R%NDGL))
IF(LLP2)WRITE(NOUT,9) 'NLOEN   ',SIZE(G%NLOEN   ),SHAPE(G%NLOEN   )

IF (G%LREDUCED_GRID) THEN
  G%NLOEN(:) = KLOEN(1:R%NDGL)
ELSE
  G%NLOEN(:) = R%NDLON
ENDIF

IF(PRESENT(LDSPLIT)) THEN
  D%LSPLIT = LDSPLIT
ENDIF

IF(PRESENT(KTMAX)) THEN
  R%NTMAX = KTMAX
ELSE
  R%NTMAX = R%NSMAX
ENDIF
IF(R%NTMAX /= R%NSMAX) THEN
  !This SHOULD work but I don't know how to test it /MH
  WRITE(NERR,*) 'R%NTMAX /= R%NSMAX',R%NTMAX,R%NSMAX
  CALL ABORT_TRANS('ESETUP_TRANS:R%NTMAX /= R%NSMAX HAS NOT BEEN VALIDATED')
ENDIF

IF(PRESENT(PWEIGHT)) THEN
  D%LWEIGHTED_DISTR = .TRUE.
  IF( D%LWEIGHTED_DISTR .AND. .NOT.D%LSPLIT )THEN
    CALL ABORT_TRANS('SETUP_TRANS: LWEIGHTED_DISTR=T AND LSPLIT=F NOT SUPPORTED')
  ENDIF
  IF(SIZE(PWEIGHT) /= SUM(G%NLOEN(:)) )THEN
    CALL ABORT_TRANS('SETUP_TRANS:SIZE(PWEIGHT) /= SUM(G%NLOEN(:))')
  ENDIF
  ALLOCATE(D%RWEIGHT(SIZE(PWEIGHT)))
  D%RWEIGHT(:)=PWEIGHT(:)
ELSE
  D%LWEIGHTED_DISTR = .FALSE.
ENDIF

IF(PRESENT(LDGRIDONLY)) THEN
  D%LGRIDONLY=LDGRIDONLY
ENDIF

IF (PRESENT(KNOEXTZL)) THEN
  R%NNOEXTZL=KNOEXTZL
ELSE
  R%NNOEXTZL=0
ENDIF

IF (PRESENT(KNOEXTZG)) THEN
  R%NNOEXTZG=KNOEXTZG
ELSE
  R%NNOEXTZG=0
ENDIF

#ifdef WITH_FFTW
IF(PRESENT(LDUSEFFTW)) THEN
  TW%LFFTW=LDUSEFFTW
ENDIF
#endif

IF(PRESENT(LDUSEFFTW)) THEN
  TALD%LFFT992=.NOT.LDUSEFFTW
ELSE
  TALD%LFFT992=.TRUE.
ENDIF

!     Setup resolution dependent structures
!     -------------------------------------

! Setup distribution independent dimensions
CALL ESETUP_DIMS
IF (PRESENT(PEXWN)) GALD%EXWN=PEXWN
IF (PRESENT(PEYWN)) GALD%EYWN=PEYWN

! First part of setup of distributed environment
CALL SUEMP_TRANS_PRELEG

CALL GSTATS(1802,0)
! Compute arrays related to grid-point geometry
CALL ESETUP_GEOM
! Second part of setup of distributed environment
CALL SUEMP_TRANS
! Initialize Fast Fourier Transform package
CALL SUEFFT
CALL GSTATS(1802,1)

! Signal the current resolution is active
LENABLED(NDEF_RESOL)=.TRUE.

IF( .NOT.D%LGRIDONLY ) THEN

WRITE(NOUT,*) '===now going to allocate GPU arrays'

!$acc enter data &
!$acc& copyin(F,F%RN,F%RLAPIN,D,D%NUMP,D%MYMS,R,R%NDGNH,R%NSMAX,G,G%NDGLU) &
!$acc& copyin(D%NPNTGTB0,D%NPNTGTB1,D%NSTAGT0B,D%NSTAGT1B,D%NSTAGTF,G%NMEN,D%NPROCM,D%NPTRLS,G,G%NLOEN,D%MSTABF)

R_NSMAX=R%NSMAX
R_NTMAX=R%NTMAX
R_NDGNH=R%NDGNH
R_NDGL=R%NDGL
R_NNOEXTZL=R%NNOEXTZL


ALLOCATE(D_NSTAGT0B(SIZE(D%NSTAGT0B)))
ALLOCATE(D_NSTAGT1B(SIZE(D%NSTAGT1B)))
ALLOCATE(D_NPNTGTB0(0:SIZE(D%NPNTGTB0,1)-1,SIZE(D%NPNTGTB0,2)))
ALLOCATE(D_NPNTGTB1(SIZE(D%NPNTGTB1,1),SIZE(D%NPNTGTB1,2)))
ALLOCATE(D_MYMS(SIZE(D%MYMS)))
ALLOCATE(D_NPROCL(SIZE(D%NPROCL)))
ALLOCATE(D_NASM0(0:SIZE(D%NASM0)-1))
ALLOCATE(D_NSTAGTF(SIZE(D%NSTAGTF)))
ALLOCATE(D_MSTABF(SIZE(D%MSTABF)))
ALLOCATE(D_NPROCM(0:SIZE(D%NPROCM)-1))
ALLOCATE(D_NPTRLS(SIZE(D%NPTRLS)))

ALLOCATE(G_NDGLU(0:SIZE(G%NDGLU)-1))
ALLOCATE(G_NMEN(SIZE(G%NMEN)))
ALLOCATE(G_NLOEN(SIZE(G%NLOEN)))

DO I=0,SIZE(G%NDGLU)-1
   G_NDGLU(I)=G%NDGLU(I)
end DO

G_NMEN_MAX=0
DO I=1,SIZE(G%NMEN)
   G_NMEN(I)=G%NMEN(I)
   if (G_NMEN(I) .gt. G_NMEN_MAX) G_NMEN_MAX=G_NMEN(I)
end DO

G_NLOEN_MAX=0
DO I=1,SIZE(G%NLOEN)
   G_NLOEN(I)=G%NLOEN(I)
   if (G_NLOEN(I) .gt. G_NLOEN_MAX) G_NLOEN_MAX=G_NLOEN(I)
end DO

DO I=1,SIZE(D%NSTAGT0B)
   D_NSTAGT0B(I)=D%NSTAGT0B(I)
END DO

DO I=1,SIZE(D%NSTAGT1B)
   D_NSTAGT1B(I)=D%NSTAGT1B(I)
END DO

DO I=1,SIZE(D%NPROCL)
   D_NPROCL(I)=D%NPROCL(I)
END DO

DO I=0,SIZE(D%NASM0)-1
   D_NASM0(I)=D%NASM0(I)
END DO

DO I=1,SIZE(D%NSTAGTF)
   D_NSTAGTF(I)=D%NSTAGTF(I)
END DO

DO I=1,SIZE(D%MSTABF)
   D_MSTABF(I)=D%MSTABF(I)
END DO

DO I=0,SIZE(D%NPROCM)-1
   D_NPROCM(I)=D%NPROCM(I)
END DO

DO I=1,SIZE(D%NPTRLS)
   D_NPTRLS(I)=D%NPTRLS(I)
END DO

DO I=1,SIZE(D%NPNTGTB0,2)
   DO J=0,SIZE(D%NPNTGTB0,1)-1
      D_NPNTGTB0(J,I)=D%NPNTGTB0(J,I)
   end DO
END DO

DO I=1,SIZE(D%NPNTGTB1,2)
   DO J=1,SIZE(D%NPNTGTB1,1)
      D_NPNTGTB1(J,I)=D%NPNTGTB1(J,I)
   end DO
END DO

D_NUMP=D%NUMP

DO I=1,SIZE(D%MYMS)
   D_MYMS(I)=D%MYMS(I)
end DO

!$ACC enter data create(R_NSMAX,R_NTMAX,R_NDGL,R_NNOEXTZL,R_NDGNH,D_NSTAGT0B,D_NSTAGT1B,D_NPNTGTB1,D_NPROCL,D_NUMP,D_MYMS,D_NASM0,D_NSTAGTF,D_MSTABF,D_NPNTGTB0,D_NPROCM,D_NPTRLS,G_NDGLU,G_NMEN,G_NMEN_MAX,G_NLOEN,G_NLOEN_MAX)

!$ACC update device(R_NSMAX,R_NTMAX,R_NDGL,R_NNOEXTZL,R_NDGNH,D_NSTAGT0B,D_NSTAGT1B,D_NPNTGTB1,D_NPROCL,D_NUMP,D_MYMS,D_NASM0,D_NSTAGTF,D_MSTABF,D_NPNTGTB0,D_NPROCM,D_NPTRLS,G_NDGLU,G_NMEN,G_NMEN_MAX,G_NLOEN,G_NLOEN_MAX)

WRITE(NOUT,*) '===GPU arrays successfully allocated'
!endif INTERFACE

ENDIF

IF (LHOOK) CALL DR_HOOK('ESETUP_TRANS',1,ZHOOK_HANDLE)
!     ------------------------------------------------------------------
9 FORMAT(1X,'ARRAY ',A10,' ALLOCATED ',8I8)

!endif INTERFACE

END SUBROUTINE ESETUP_TRANS

