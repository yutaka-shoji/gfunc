# gfunc
Python package to evaluate g-function for ground heat exchanger simulation.
Following models is implemented in the latest version.

- ILS: Infinite Line Source model
- ICS: Infinite Cylinder Source model
- CM-ILS: Composite-Medium Infinite Line Source model
- MILS: Moving Infinite Line Source model
- MICS: Moving Infinite Cylinder Source model (ANN implementation)

## Quick-start
You can try in your browser with Google Colab. <br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutaka-shoji/gfunc/blob/main/quickstart.ipynb)

## Dependencies
Before install this package, you need to install a fortran compiler such as [gfortran](https://gcc.gnu.org/wiki/GFortran).

## Installation
```sh
pip install git+https://github.com/yutaka-shoji/gfunc
```

## References
- ILS, ICS: Ingersoll, L.R., Zobel, O.J., Ingersoll, A.C., 1948. Heat conduction, with engineering and geological applications. McGraw-Hill Book Co.
- ILS, ICS: Carslaw, H.S., Jaeger, J.C., 1959. Conduction of heat in solids, Oxford: Clarendon Press, 1959, 2nd ed.
- MILS: Diao, N., Li, Q., Fang, Z., 2004. Heat transfer in ground heat exchangers with groundwater advection. International Journal of Thermal Sciences 43, 1203–1211. https://doi.org/10.1016/j.ijthermalsci.2004.04.009
- CM-ILS: Li, M., Lai, A.C.K., 2012. New temperature response functions (G functions) for pile and borehole ground heat exchangers based on composite-medium line-source theory. Energy 38, 255–263. https://doi.org/10.1016/j.energy.2011.12.004
- MICS: Shoji, Y., Katsura, T., Nagano, K., 2022. MICS-ANN model: An artificial neural network model for fast computation of G-function in moving infinite cylindrical source model. Geothermics 100, 102315. https://doi.org/10.1016/j.geothermics.2021.102315


MIT license
