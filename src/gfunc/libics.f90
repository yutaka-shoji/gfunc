subroutine integrand(x, Fo, R, f)
  implicit none
  double precision, intent(in) :: x
  double precision, intent(in) :: Fo
  double precision, intent(in) :: R
  double precision, intent(out) :: f
  !-------------------------------------------------------------------------
  f = ( exp( - x*x * Fo ) - 1 ) &
    * ( bessel_j0( x * R ) * bessel_y1( x ) - bessel_y0( x * R ) * bessel_j1( x ) )&
    / ( x*x * ( bessel_j1( x ) ** 2 + bessel_y1( x ) ** 2 ) )

  return
end subroutine integrand
