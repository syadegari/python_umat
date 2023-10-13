subroutine GRAINORIENTATIONBCC(SlipSys, RM)
  !
  !  declaration of variables
  double precision SlipSys(24,3,3)
  double precision RM(3,3)
  double precision X(3,3), Temp
  integer i,j,k,a,b
  !
  !  calculate the rotated slip system tensors
  do k = 1,24
     do i = 1,3
        do j = 1,3
           X(i,j) = SlipSys(k,i,j)
        enddo
     enddo
     do i = 1,3
        do j = 1,3
           Temp = 0.d0
           do a = 1,3
              do b = 1,3
                 Temp = Temp + X(a,b)*RM(i,a)*RM(j,b)
              enddo
           enddo
           SlipSys(k,i,j)  = Temp
        enddo
     enddo
  enddo
  !
  !    ..return the rotated Schmid-tensors
  return
end subroutine GRAINORIENTATIONBCC


program RotateFortran
  implicit none
  double precision :: RM(3, 3)
  double precision :: SlipSys(24, 3, 3)
  integer :: i, j

  open(10, file='rotation.dat')
  open(20, file='slipsys.dat')
  open(30, file='output_fortran.dat')

  ! Read rotation matrix and slip systems
  do i = 1,3
     read(10, *) RM(i, :)
  enddo
  do i = 1,24
     do j = 1,3
        read(20, *) SlipSys(i, j, :)
     enddo
  enddo

  ! Rotate slip systems
  call GRAINORIENTATIONBCC(SlipSys, RM)

  ! Write results to file
  do i = 1,24
     do j = 1,3
        write(30, *) SlipSys(i, j, :)
     enddo
  enddo

  close(10)
  close(20)
  close(30)
end program RotateFortran
