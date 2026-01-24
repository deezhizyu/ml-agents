"""
Tests for shared memory environment manager
"""
import pytest
import numpy as np
from mlagents.trainers.env_manager_shared_memory import SharedMemoryBuffer


class TestSharedMemoryBuffer:
    """Test SharedMemoryBuffer class"""
    
    def test_buffer_creation(self):
        """Test creating a shared memory buffer"""
        shape = (10, 5)
        dtype = np.float32
        
        buffer = SharedMemoryBuffer("test_buffer_1", shape, dtype)
        
        try:
            assert buffer.name == "test_buffer_1"
            assert buffer.shape == shape
            assert buffer.dtype == dtype
            assert buffer.array.shape == shape
            assert buffer.array.dtype == dtype
        finally:
            buffer.close()
            buffer.unlink()
    
    def test_buffer_write_read(self):
        """Test writing and reading from buffer"""
        shape = (5, 3)
        dtype = np.float32
        
        buffer = SharedMemoryBuffer("test_buffer_2", shape, dtype)
        
        try:
            # Write data
            test_data = np.random.randn(*shape).astype(dtype)
            buffer.write(test_data)
            
            # Read data
            read_data = buffer.read()
            
            # Should be same data (zero-copy)
            np.testing.assert_array_equal(read_data, test_data)
            
        finally:
            buffer.close()
            buffer.unlink()
    
    def test_buffer_zero_copy(self):
        """Test that buffer uses zero-copy (views same memory)"""
        shape = (3, 3)
        dtype = np.float32
        
        buffer = SharedMemoryBuffer("test_buffer_3", shape, dtype)
        
        try:
            # Write initial data
            data1 = np.ones(shape, dtype=dtype)
            buffer.write(data1)
            
            # Read data
            read_data = buffer.read()
            
            # Modify through buffer's array
            buffer.array[0, 0] = 99.0
            
            # Read again - should see the change
            read_data_2 = buffer.read()
            assert read_data_2[0, 0] == 99.0
            
        finally:
            buffer.close()
            buffer.unlink()
    
    def test_buffer_shape_mismatch(self):
        """Test that writing wrong shape raises error"""
        shape = (5, 5)
        dtype = np.float32
        
        buffer = SharedMemoryBuffer("test_buffer_4", shape, dtype)
        
        try:
            wrong_shape_data = np.zeros((3, 3), dtype=dtype)
            
            with pytest.raises(ValueError, match="Shape mismatch"):
                buffer.write(wrong_shape_data)
                
        finally:
            buffer.close()
            buffer.unlink()
    
    def test_buffer_cleanup(self):
        """Test proper cleanup of shared memory"""
        shape = (2, 2)
        dtype = np.float32
        
        buffer = SharedMemoryBuffer("test_buffer_5", shape, dtype)
        
        # Should be able to close and unlink
        buffer.close()
        buffer.unlink()
        
        # After cleanup, should be able to create new buffer with same name
        buffer2 = SharedMemoryBuffer("test_buffer_5", shape, dtype)
        try:
            assert buffer2.name == "test_buffer_5"
        finally:
            buffer2.close()
            buffer2.unlink()
    
    def test_buffer_with_different_dtypes(self):
        """Test buffers with different data types"""
        shape = (4, 4)
        
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            buffer_name = f"test_buffer_dtype_{dtype.__name__}"
            buffer = SharedMemoryBuffer(buffer_name, shape, dtype)
            
            try:
                test_data = np.random.randn(*shape).astype(dtype)
                buffer.write(test_data)
                read_data = buffer.read()
                
                np.testing.assert_array_equal(read_data, test_data)
                assert read_data.dtype == dtype
                
            finally:
                buffer.close()
                buffer.unlink()
    
    def test_buffer_large_arrays(self):
        """Test buffer with larger arrays (visual observations)"""
        # Simulate 84x84x3 visual observation (common size)
        shape = (84, 84, 3)
        dtype = np.uint8
        
        buffer = SharedMemoryBuffer("test_buffer_large", shape, dtype)
        
        try:
            # Create random "image"
            test_image = np.random.randint(0, 255, shape, dtype=dtype)
            buffer.write(test_image)
            
            read_image = buffer.read()
            np.testing.assert_array_equal(read_image, test_image)
            
        finally:
            buffer.close()
            buffer.unlink()
    
    def test_buffer_concurrent_access(self):
        """Test that buffer can be accessed from multiple places"""
        shape = (3, 3)
        dtype = np.float32
        buffer_name = "test_buffer_concurrent"
        
        # Create first buffer
        buffer1 = SharedMemoryBuffer(buffer_name, shape, dtype)
        
        try:
            # Write data through first buffer
            test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
            buffer1.write(test_data)
            
            # Create second buffer with same name (simulates another process)
            buffer2 = SharedMemoryBuffer(buffer_name, shape, dtype)
            
            try:
                # Read from second buffer - should see same data
                read_data = buffer2.read()
                np.testing.assert_array_equal(read_data, test_data)
                
            finally:
                buffer2.close()
                
        finally:
            buffer1.close()
            buffer1.unlink()


class TestSharedMemoryIntegration:
    """Integration tests for shared memory functionality"""
    
    def test_multiple_buffers(self):
        """Test managing multiple shared memory buffers"""
        buffers = []
        
        try:
            # Create multiple buffers
            for i in range(5):
                buffer = SharedMemoryBuffer(
                    f"test_multi_buffer_{i}",
                    (10, 10),
                    np.float32
                )
                buffers.append(buffer)
                
                # Write unique data to each
                data = np.full((10, 10), i, dtype=np.float32)
                buffer.write(data)
            
            # Verify each buffer has correct data
            for i, buffer in enumerate(buffers):
                read_data = buffer.read()
                expected = np.full((10, 10), i, dtype=np.float32)
                np.testing.assert_array_equal(read_data, expected)
                
        finally:
            # Cleanup all buffers
            for buffer in buffers:
                buffer.close()
                buffer.unlink()
    
    def test_buffer_reuse(self):
        """Test reusing buffer multiple times"""
        shape = (5, 5)
        dtype = np.float32
        buffer_name = "test_buffer_reuse"
        
        buffer = SharedMemoryBuffer(buffer_name, shape, dtype)
        
        try:
            # Write and read multiple times
            for i in range(10):
                data = np.full(shape, i, dtype=dtype)
                buffer.write(data)
                
                read_data = buffer.read()
                np.testing.assert_array_equal(read_data, data)
                
        finally:
            buffer.close()
            buffer.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
