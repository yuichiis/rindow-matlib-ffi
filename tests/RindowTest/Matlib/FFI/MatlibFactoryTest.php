<?php
namespace RindowTest\Matlib\FFI\MatlibFactoryTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Buffer\FFI\Buffer;
use Rindow\Matlib\FFI\MatlibFactory;
use Rindow\Matlib\FFI\Matlib;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use OutOfRangeException;
use TypeError;
use ArrayObject;
use ArrayAccess;
use SplFixedArray;

class MatlibFactoryTest extends TestCase
{
    public function testConfig()
    {
        $factory = new MatlibFactory();
        $this->assertTrue($factory->isAvailable());
    }
}
