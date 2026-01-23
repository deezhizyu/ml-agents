using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Manages batched observation collection for improved performance.
    /// This class pre-allocates buffers and provides pooling for observation data
    /// to reduce GC pressure during training and inference.
    /// </summary>
    internal class BatchedObservationManager : IDisposable
    {
        /// <summary>
        /// Default capacity for observation buffers.
        /// </summary>
        const int k_DefaultBufferCapacity = 512;

        /// <summary>
        /// Pool of float arrays for observation data.
        /// </summary>
        readonly Stack<float[]> m_FloatArrayPool = new Stack<float[]>(32);

        /// <summary>
        /// Pool of List&lt;float&gt; for variable-size observations.
        /// </summary>
        readonly Stack<List<float>> m_FloatListPool = new Stack<List<float>>(32);

        /// <summary>
        /// Tracks arrays currently in use (for debugging/validation).
        /// </summary>
        int m_ArraysInUse = 0;

        /// <summary>
        /// Statistics for monitoring pool efficiency.
        /// </summary>
        int m_TotalAllocations = 0;
        int m_PoolHits = 0;

        bool m_Disposed = false;

        /// <summary>
        /// Gets a float array from the pool or allocates a new one.
        /// </summary>
        /// <param name="size">Required size of the array.</param>
        /// <returns>A float array of at least the requested size.</returns>
        public float[] RentFloatArray(int size)
        {
            Profiler.BeginSample("BatchedObservationManager.RentFloatArray");

            float[] array = null;

            // Try to find a suitable array in the pool
            if (m_FloatArrayPool.Count > 0)
            {
                array = m_FloatArrayPool.Pop();
                if (array.Length < size)
                {
                    // Array too small, return to pool and allocate new
                    m_FloatArrayPool.Push(array);
                    array = null;
                }
                else
                {
                    m_PoolHits++;
                }
            }

            if (array == null)
            {
                // Allocate with some extra capacity to reduce future allocations
                var capacity = Math.Max(size, k_DefaultBufferCapacity);
                array = new float[capacity];
                m_TotalAllocations++;
            }

            m_ArraysInUse++;

            Profiler.EndSample();
            return array;
        }

        /// <summary>
        /// Returns a float array to the pool for reuse.
        /// </summary>
        /// <param name="array">The array to return.</param>
        public void ReturnFloatArray(float[] array)
        {
            if (array == null) return;

            Profiler.BeginSample("BatchedObservationManager.ReturnFloatArray");

            m_ArraysInUse--;

            // Only keep arrays up to a certain size to avoid memory bloat
            if (array.Length <= k_DefaultBufferCapacity * 4)
            {
                m_FloatArrayPool.Push(array);
            }

            Profiler.EndSample();
        }

        /// <summary>
        /// Gets a List&lt;float&gt; from the pool or allocates a new one.
        /// </summary>
        /// <param name="capacity">Initial capacity for the list.</param>
        /// <returns>A List&lt;float&gt; with at least the requested capacity.</returns>
        public List<float> RentFloatList(int capacity)
        {
            List<float> list = null;

            if (m_FloatListPool.Count > 0)
            {
                list = m_FloatListPool.Pop();
                list.Clear();
                if (list.Capacity < capacity)
                {
                    list.Capacity = capacity;
                }
                m_PoolHits++;
            }
            else
            {
                list = new List<float>(capacity);
                m_TotalAllocations++;
            }

            return list;
        }

        /// <summary>
        /// Returns a List&lt;float&gt; to the pool for reuse.
        /// </summary>
        /// <param name="list">The list to return.</param>
        public void ReturnFloatList(List<float> list)
        {
            if (list == null) return;

            list.Clear();

            // Only keep lists up to a certain capacity
            if (list.Capacity <= k_DefaultBufferCapacity * 4)
            {
                m_FloatListPool.Push(list);
            }
        }

        /// <summary>
        /// Clears all pooled arrays. Call this when changing scenes or resetting.
        /// </summary>
        public void ClearPools()
        {
            m_FloatArrayPool.Clear();
            m_FloatListPool.Clear();
        }

        /// <summary>
        /// Gets statistics about pool usage for debugging.
        /// </summary>
        /// <returns>A string describing pool statistics.</returns>
        public string GetPoolStatistics()
        {
            var hitRate = m_TotalAllocations > 0
                ? (float)m_PoolHits / (m_PoolHits + m_TotalAllocations) * 100f
                : 0f;

            return $"BatchedObservationManager Stats: " +
                   $"ArraysInUse={m_ArraysInUse}, " +
                   $"PooledArrays={m_FloatArrayPool.Count}, " +
                   $"PooledLists={m_FloatListPool.Count}, " +
                   $"HitRate={hitRate:F1}%";
        }

        public void Dispose()
        {
            if (m_Disposed) return;

            ClearPools();
            m_Disposed = true;
        }
    }
}
