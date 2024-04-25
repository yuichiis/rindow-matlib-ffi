<?php
namespace Rindow\Matlib\FFI;

use FFI;
//use FFI\Env\Runtime as FFIEnvRuntime;
//use FFI\Env\Status as FFIEnvStatus;
//use FFI\Location\Locator as FFIEnvLocator;
use FFI\Exception as FFIException;
use RuntimeException;

class MatlibFactory
{
    private static ?FFI $ffi = null;
    private static ?string $libFile = null;

    /** @var array<string> $libs_win */
    protected array $libs_win = ['rindowmatlib.dll'];
    /** @var array<string> $libs_linux */
    protected array $libs_linux = ['librindowmatlib.so'];
    /** @var array<string> $libs_mac */
    protected array $libs_mac = ['librindowmatlib.dylib', '/usr/local/lib/librindowmatlib.dylib', '/usr/lib/librindowmatlib.dylib'];
    protected ?string $error = null;

    /** @param array<string> $libFiles */
    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        if(!extension_loaded('ffi')) {
            return;
        }
        $headerFile = $headerFile ?? __DIR__.'/matlib.h';
        if($libFiles==null) {
            if(PHP_OS=='Linux') {
                $libFiles = $this->libs_linux;
            } elseif(PHP_OS=='Darwin') {
                $libFiles = $this->libs_mac;
            } elseif(PHP_OS=='WINNT') {
                $libFiles = $this->libs_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($headerFile);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //$pathname = FFIEnvLocator::resolve(...$libFiles);
        //if($pathname) {
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffi = $ffi;
        //}
        foreach ($libFiles as $filename) {
            try {
                $ffi = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                $this->error = "$filename not found";
                continue;
            }
            $this->error = null;
            self::$ffi = $ffi;
            self::$libFile = $filename;
            break;
        }
        //if(PHP_OS=='Linux') {
        //    if(self::$ffi!==null) {
        //        $mode = self::$ffi->rindow_matlib_common_get_parallel();
        //        if($mode==Matlib::P_OPENMP) {
        //            throw new RuntimeException('OpenMP does not work properly in the Linux version of PHP. Please switch to serial version of matlib.');
        //        }
        //    }
        //}
    }

    public function isAvailable() : bool
    {
        return self::$ffi!==null;
        //$isAvailable = FFIEnvRuntime::isAvailable();
        //if(!$isAvailable) {
        //    return false;
        //}
        //$pathname = FFIEnvLocator::resolve(...$this->libs);
        //return $pathname!==null;
    }

    public function Matlib() : Matlib
    {
        if(self::$ffi==null) {
            throw new RuntimeException('rindow-matlib library not loaded.');
        }
        return new Matlib(self::$ffi);
    }

    public function Math() : Matlib
    {
        return $this->Matlib();
    }

    /** @return array<string,mixed> */
    public function config() : array
    {
        return [
            'OS' => PHP_OS,
            'FFI' => class_exists(FFI::class),
            'libFile' => self::$libFile,
            'error' => $this->error,
        ];
    }
}
