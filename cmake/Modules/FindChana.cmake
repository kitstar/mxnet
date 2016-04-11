SET(ChaNa_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/chana  
  /usr/local/include
  /usr/local/include/chana  
  /opt/chana/include
  ${PROJECT_SOURCE_DIR}/3rdparty/chana/include
  ${PROJECT_SOURCE_DIR}/thirdparty/chana/include
  $ENV{ChaNa_HOME}
  $ENV{ChaNa_HOME}/include
)

SET(ChaNa_LIB_SEARCH_PATHS
        /lib/
        /lib/chana
        /lib64/
        /usr/lib
        /usr/lib/chana
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/chana/lib
        ${PROJECT_SOURCE_DIR}/3rdparty/chana/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/chana/lib        
        $ENV{ChaNa_HOME}
        $ENV{ChaNa_HOME}/lib
 )

FIND_PATH(ChaNa_INCLUDE_DIR NAMES chana_ps.h PATHS ${ChaNa_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(ChaNa_LIB NAMES chanalib.lib PATHS ${ChaNa_LIB_SEARCH_PATHS})

SET(ChaNa_FOUND ON)

#    Check include files
IF(NOT ChaNa_INCLUDE_DIR)
    SET(ChaNa_FOUND OFF)
    MESSAGE(STATUS "Could not find ChaNa include. Turning ChaNa_FOUND off")
ENDIF()

#    Check libraries
IF(NOT ChaNa_LIB)
    SET(ChaNa_FOUND OFF)
    MESSAGE(STATUS "Could not find ChaNa lib. Turning ChaNa_FOUND off")
ENDIF()

IF (ChaNa_FOUND)
  IF (NOT ChaNa_FIND_QUIETLY)
    MESSAGE(STATUS "Found ChaNa libraries: ${ChaNa_LIB}")
    MESSAGE(STATUS "Found ChaNa include: ${ChaNa_INCLUDE_DIR}")
  ENDIF (NOT ChaNa_FIND_QUIETLY)
ELSE (ChaNa_FOUND)
  IF (ChaNa_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ChaNa")
  ENDIF (ChaNa_FIND_REQUIRED)
ENDIF (ChaNa_FOUND)

MARK_AS_ADVANCED(
    ChaNa_INCLUDE_DIR
    ChaNa_LIB    
)