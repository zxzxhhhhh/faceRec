cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(face_recognition)

# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# it needs is the dlib source code folder and it will take care of everything.
include(../dlib/cmake)


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the face_recognition example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})
      add_executable(face_recognition face_recognition.cpp face_gui.hpp)
      target_link_libraries(face_recognition dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the face_recognition example.")
   endif()
endif()

