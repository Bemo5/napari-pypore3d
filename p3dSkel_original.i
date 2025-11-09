%module p3dSkel

%{
  #define SWIG_FILE_WITH_INIT
  #define PY_ARRAY_UNIQUE_SYMBOL PyPore3D_ARRAY_API_SKEL
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <float.h>
  #include <assert.h>

  // Project headers brought into the C preamble
  #include "P3D_Skel/p3dSkel.h"
  #include "P3D_Skel/p3dTime.h"
  #include "P3D_Skel/Common/p3dBoundingBoxT.h"
  #include "P3D_Skel/Common/p3dConnectedComponentsLabeling.h"
  #include "P3D_Skel/Common/p3dCoordsList.h"
  #include "P3D_Skel/Common/p3dCoordsQueue.h"
  #include "P3D_Skel/Common/p3dCoordsT.h"
  #include "P3D_Skel/Common/p3dFCoordsList.h"
  #include "P3D_Skel/Common/p3dUIntList.h"
  #include "P3D_Skel/Common/p3dUtils.h"
%}

%include "typemaps.i"
%include "exception.i"
%include "numpy.i"

%init %{
  import_array();
%}

%include "cpointer.i"
%inline %{
  typedef struct SkeletonStats SkeletonStats;
%}
%pointer_class(SkeletonStats, SkeletonStats *);

/* -------------------------------------------
   Include project headers BEFORE allocators
   (already brought in via %{...%}, now expose to SWIG interface)
   ------------------------------------------- */
%include "P3D_Skel/p3dTime.h"
%include "P3D_Skel/Common/p3dBoundingBoxT.h"
%include "P3D_Skel/Common/p3dConnectedComponentsLabeling.h"
%include "P3D_Skel/Common/p3dCoordsList.h"
%include "P3D_Skel/Common/p3dCoordsQueue.h"
%include "P3D_Skel/Common/p3dCoordsT.h"
%include "P3D_Skel/Common/p3dFCoordsList.h"
%include "P3D_Skel/Common/p3dUIntList.h"
%include "P3D_Skel/Common/p3dUtils.h"


