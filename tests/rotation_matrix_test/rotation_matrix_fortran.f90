subroutine MULTIMATRIX3X3(Ma,Mb,Mc)
  !
  !     declaration of variables
  double precision Ma(3,3),Mb(3,3),Mc(3,3)
  integer i,j,k
  !
  !     compute the multiplication of matrices
  do i = 1,3
     do j = 1,3
        Mc(i,j) = 0.d0
        do k = 1,3
           Mc(i,j) = Mc(i,j) + Ma(i,k)*Mb(k,j)
        enddo
     enddo
  enddo
  !
  return
end subroutine MULTIMATRIX3X3


subroutine ROTATIONMATRIX(Angle1,Angle2,Angle3,RM)
  !
  !  declaration of variables
  double precision RM(3,3),R1(3,3),R2(3,3),R3(3,3),Angle1,Angle2,Angle3,Temp(3,3)
  !
  !  rotation matrix of the first rotation (Angle1)
  R1 = 0.d0
  R1(1,1) =  cos(Angle1)
  R1(1,2) =  sin(Angle1)
  R1(2,1) = -sin(Angle1)
  R1(2,2) =  cos(Angle1)
  R1(3,3) =  1.d0
  !
  !  rotation matrix of the second rotation (Angle2)
  R2 = 0.d0
  R2(1,1) =  cos(Angle2)
  R2(1,3) = -sin(Angle2)
  R2(2,2) =  1.d0
  R2(3,1) =  sin(Angle2)
  R2(3,3) =  cos(Angle2)
  !
  !  rotation matrix of the third rotation (Angle3)
  R3 = 0.d0
  R3(1,1) =  cos(Angle3)
  R3(1,2) =  sin(Angle3)
  R3(2,1) = -sin(Angle3)
  R3(2,2) =  cos(Angle3)
  R3(3,3) =  1.d0
  !
  !  calculate the overall rotation matrix
  call MULTIMATRIX3X3(R3,R2,Temp)
  call MULTIMATRIX3X3(Temp,R1,RM)
  !
  return
end subroutine ROTATIONMATRIX


program RotateMatrixFortran
  implicit none
  double precision :: Angles(3)
  double precision :: RM(3, 3)
  integer :: i

  open(10, file='angles.dat')
  open(20, file='rotation_matrix_fortran.dat')

  read(10, *) Angles(:)
  call ROTATIONMATRIX(Angles(1), Angles(2), Angles(3), RM)

  do i = 1, 3
     write(20, *) RM(i, :)
  enddo

  close(10)
  close(20)
end program RotateMatrixFortran
