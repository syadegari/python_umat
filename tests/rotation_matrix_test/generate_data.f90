subroutine AssignRotationAngles(angles)
  implicit none
  double precision, intent(out) :: angles(3)

  ! Content from slip_system.f90 goes here
  include 'orientation.f90'
end subroutine AssignRotationAngles

program GenerateData
  implicit none
  double precision :: Angles(3)
  ! This is the angles file. We will read from this in both
  ! Fortran and Python version of rotation function.
  open(10, file='angles.dat')

  ! Load the angles
  call AssignRotationAngles(Angles)

  ! Write the angels
  write(10, *) Angles(:)

  close(10)
end program GenerateData
