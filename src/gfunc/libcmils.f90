subroutine integrand(x, n, Fo, RA, RB, RD, alpha, k, func)
  implicit none
  double precision, intent(in) :: x
  integer, intent(in) :: n
  double precision, intent(in) :: Fo, RA, RB, RD, alpha, k
  double precision, intent(out) :: func

  double precision :: phi, psi, f, g
  double precision :: jn_x, jnp_x, jn_ax, jnp_ax
  double precision :: yn_x, ynp_x, yn_ax, ynp_ax
  double precision :: ax
  !-------------------------------------------------------------------------
  ax = alpha * x

  jn_x  = bessel_jn(n, x)
  jnp_x = n/x * jn_x - bessel_jn(n+1, x)
  jn_ax  = bessel_jn(n, ax)
  jnp_ax = n/ax * jn_ax - bessel_jn(n+1, ax)
  yn_x  = bessel_yn(n, x)
  ynp_x = n/x * yn_x - bessel_yn(n+1, x)
  yn_ax  = bessel_yn(n, ax)
  ynp_ax = n/ax * yn_ax - bessel_yn(n+1, ax)

  phi = alpha * k * jn_x * jnp_ax - jnp_x * jn_ax
  psi = alpha * k * jn_x * ynp_ax - jnp_x * yn_ax
  f   = alpha * k * yn_x * jnp_ax - ynp_x * jn_ax
  g   = alpha * k * yn_x * ynp_ax - ynp_x * yn_ax

  func = ( 1.0d0 - exp(-x*x * Fo) ) &
    * ( bessel_jn(n, RA*x) + bessel_jn(n, RB*x) ) * 0.5d0 &
    * bessel_jn(n, RD*x) &
    * ( phi * g - psi * f ) / ( x * ( phi*phi + psi*psi ) )

  return
end subroutine integrand
