The Interface of The Rindow Matlib on FFI
==========================================

Status:
[![Build Status](https://github.com/rindow/rindow-matlib-ffi/workflows/tests/badge.svg)](https://github.com/rindow/rindow-matlib-ffi/actions)
[![Downloads](https://img.shields.io/packagist/dt/rindow/rindow-matlib-ffi)](https://packagist.org/packages/rindow/rindow-matlib-ffi)
[![Latest Stable Version](https://img.shields.io/packagist/v/rindow/rindow-matlib-ffi)](https://packagist.org/packages/rindow/rindow-matlib-ffi)
[![License](https://img.shields.io/packagist/l/rindow/rindow-matlib-ffi)](https://packagist.org/packages/rindow/rindow-matlib-ffi)

"The interface of rindow matlib on ffi".

Please see the documents about rindow mathematics on [Rindow Mathematics](https://rindow.github.io/mathematics/openblas/mathlibrary.html) web pages.

You can call a high-speed calculation library written in C language to speed up matrix calculation processing.
Rindow Matlib includes many matrix operations functions used in machine learning.

Requirements
============

- PHP 8.1 or PHP8.2 or PHP8.3 or PHP8.4
- Rindow Matlib C Library 1.1 or later
- Windows 10/11, Linux, macOS


How to setup
============

### How to setup for Windows
Download the pre-build binary file.

- https://github.com/rindow/rindow-matlib/releases

Unzip the file for Windows and copy rindowmatlib.dll to the directory set in PATH.

The standard DLLs in the Bin directory is the thread version. Subdirectories contain OpenMP and Serial versions.

```shell
C> copy rindowmatlib.dll C:\php
C> PATH %PATH%;C:\php
```

Set it up using composer.

```shell
C> mkdir \your\app\dir
C> cd \your\app\dir
C> composer require rindow/rindow-matlib-ffi
```

### How to setup for Linux
Download the pre-build binary file.

- https://github.com/rindow/rindow-matlib/releases

Please install using the apt command. 
```shell
$ sudo apt install ./rindow-matlib_X.X.X_amd64.deb
```

Set it up using composer.

```shell
$ mkdir \your\app\dir
$ cd \your\app\dir
$ composer require rindow/rindow-matlib-ffi
```

### Troubleshooting for Linux
Since rindow-matlib currently uses ptheads, so you should choose the pthread version for OpenBLAS as well.
In version 1.0 of Rindow-matlib we recommended the OpenMP version, but now we have changed our policy and are recommending the pthread version.

Using the OpenMP version of OpenBLAS can cause conflicts and become unstable and slow.
This issue does not occur on Windows.

If you have already installed the OpenMP version of OpenBLAS, you can delete it and install pthread version.
```shell
$ sudo apt install libopenblas0-pthread liblapacke
$ sudo apt remove libopenblas0-openmp
```

But if you can't remove it, you can switch to it using the update-alternatives command.

```shell
$ sudo update-alternatives --config libopenblas.so.0-x86_64-linux-gnu
$ sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu
```

If you really want to use the OpenMP version of OpenBLAS, please switch to the OpenMP version of rindow-matlib.

```shell
$ sudo update-alternatives --config librindowmatlib.so
There are 1 choices for the alternative librindowmatlib.so (providing /usr/lib/librindowmatlib.so).

  Selection    Path                                             Priority   Status
------------------------------------------------------------
* 0            /usr/lib/rindowmatlib-thread/librindowmatlib.so   95        auto mode
  1            /usr/lib/rindowmatlib-openmp/librindowmatlib.so   95        manual mode
  2            /usr/lib/rindowmatlib-serial/librindowmatlib.so   90        manual mode
  3            /usr/lib/rindowmatlib-thread/librindowmatlib.so   100       manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
```
Choose the "rindowmatlib-openmp".



How to use
==========

```shell
$ composer require rindow/rindow-matlib-ffi
$ composer require rindow/rindow-math-buffer-ffi
```

```php
<?php
include __DIR__.'/vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Matlib\FFI\MatlibFactory;
use Rindow\Math\Buffer\FFI\BufferFactory;

$factory = new MatlibFactory();
$math = $factory->Math();
$hostBufferFactory = new BufferFactory();
$NWITEMS = 64;

$x = $hostBufferFactory->Buffer(
    $NWITEMS,NDArray::float32
);

for($i=0;$i<$NWITEMS;$i++) {
    $x[$i] = $i;
}
$n = count($x);
$offsetX = 0;
$incX = 1;

$sum = $math->sum($n,$x,$offsetX,$incX);

var_dump($sum);
```
