- 👋 Hi, I’m @BWMa688
- 👀 I’m interested in Machine learning in Materials Science and data analysis, SISSO.
- 🌱 I’m currently learning symbolic regression in perovskite ceramics. Here will be my usual use of some code, welcome to discuss with me.
- 📫 How to reach me：mabowen.sust@qq.com 

Another symbolic regression algorithm SISSO is related to：https://github.com/rouyang2017/SISSO

Installation
-------------
A Fortran mpi compiler is required to compile the SISSO parallel program. Below are two options for compiling the program using an IntelMPI compiler (other compilers may work as well). In the folder 'src', do:    
(1)  mpiifort -fp-model precise var_global.f90 libsisso.f90 DI.f90 FC.f90 SISSO.f90 -o ~/bin/your_code_name  
(2)  mpiifort -O2 var_global.f90 libsisso.f90 DI.f90 FC.f90 SISSO.f90 -o ~/bin/your_code_name  
  


<!---
BWMa688/BWMa688 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
