rem NEED to set the path to the 64 bit tools, else it will crash!
set path=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;%path%


where cl.exe >nul 2>nul && goto WHERE_CL_END
  rem Find VC - https://github.com/microsoft/vswhere/wiki/Find-VC#batch
  set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if not exist "%vswhere%" (
    echo ERROR - Failed to find "vswhere.exe".  Please install Visual Studio. && goto :END
  )

  for /f "usebackq tokens=*" %%i in (
    `"%vswhere%" -latest ^
                 -products * ^
                 -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
                 -property installationPath`
  ) do (
    set "InstallDir=%%i"
  )

  if not exist "%InstallDir%\Common7\Tools\vsdevcmd.bat" (
    echo ERROR - Failed to find "vsdevcmd.bat".  Please install Visual C++. && goto :END
  )
  call "%InstallDir%\Common7\Tools\vsdevcmd.bat" -arch=x64 -host_arch=x64 || goto :END
:WHERE_CL_END

rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum spectrum.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-toroidal gbn-toroidal.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-adaptive gbn-adaptive.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-reconstruct gbn-reconstruct.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-bounded gbn-bounded.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum-nd spectrum-nd.cu -lcairo

nvcc -o spectrum spectrum.cu -lcairo
nvcc -o gbn-toroidal gbn-toroidal.cu -lcairo
nvcc -o gbn-adaptive gbn-adaptive.cu -lcairo
nvcc -o gbn-reconstruct gbn-reconstruct.cu -lcairo
nvcc -o gbn-bounded gbn-bounded.cu -lcairo
nvcc -o spectrum-nd spectrum-nd.cu -lcairo

:END
