<?php
namespace Rindow\Matlib\FFI;

use FFI;
use FFI\Env\Runtime as FFIEnvRuntime;
use FFI\Env\Status as FFIEnvStatus;
use FFI\Location\Locator as FFIEnvLocator;

class MatlibFactory
{
    private static ?FFI $ffi = null;
    protected array $libs = ['libmatlib.so','matlib.dll'];

    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        $headerFile = $headerFile ?? __DIR__ . "/matlib_win.h";
        $libFiles = $libFiles ?? $this->libs;
        $code = file_get_contents($headerFile);
        $pathname = FFIEnvLocator::resolve(...$libFiles);
        if($pathname) {
            $ffi = FFI::cdef($code,$pathname);
            self::$ffi = $ffi;
        }
    }

    public function isAvailable() : bool
    {
        $isAvailable = FFIEnvRuntime::isAvailable();
        if(!$isAvailable) {
            return false;
        }
        $pathname = FFIEnvLocator::resolve(...$this->libs);
        return $pathname!==null;
    }

    public function Matlib() : Matlib
    {
        return new Matlib(self::$ffi);
    }

    public function Math() : Matlib
    {
        return $this->Matlib();
    }

    public function config() : void
    {
        $isAvailable = FFIEnvRuntime::isAvailable();
        //var_dump($isAvailable);
        $lib = 'matlib.dll';
        //$lib = 'libOpenCL.so';
        $exists = FFIEnvLocator::exists($lib);
        //echo "exists:"; var_dump($exists);
        $pathname = FFIEnvLocator::pathname($lib);
        //echo "pathname:"; var_dump($pathname);
        $pathname = FFIEnvLocator::resolve('test.so', 'libvulkan.so');//, $lib);
        //echo "resolve:"; var_dump($pathname);
    }
}
