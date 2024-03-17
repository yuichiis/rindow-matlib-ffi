<?php
namespace RindowTest\Matlib\FFI\ReleaseTest;

use PHPUnit\Framework\TestCase;
use Rindow\Matlib\FFI\MatlibFactory;
use Rindow\Matlib\FFI\Matlib;
use FFI;

class ReleaseTest extends TestCase
{
    public function testFFINotLoaded()
    {
        $factory = new MatlibFactory();
        if(extension_loaded('ffi')) {
            $math = $factory->Math();
            $this->assertInstanceof(Matlib::class,$math);
        } else {
            $this->assertFalse($factory->isAvailable());
        }
    }
}