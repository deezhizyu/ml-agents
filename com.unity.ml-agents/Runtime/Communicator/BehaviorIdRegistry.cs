#if UNITY_EDITOR || UNITY_STANDALONE
using System.Collections.Generic;

namespace Unity.MLAgents
{
    /// <summary>
    /// Registry that maps behavior name strings to integer IDs for faster dictionary lookups.
    /// Integer hashing is significantly faster than string hashing.
    /// </summary>
    public class BehaviorIdRegistry
    {
        private readonly Dictionary<string, int> m_NameToId = new Dictionary<string, int>();
        private readonly List<string> m_IdToName = new List<string>();
        private readonly object m_Lock = new object();

        /// <summary>
        /// Gets or creates an integer ID for the given behavior name.
        /// Thread-safe.
        /// </summary>
        /// <param name="behaviorName">The behavior name string.</param>
        /// <returns>The integer ID for this behavior.</returns>
        public int GetOrCreateId(string behaviorName)
        {
            lock (m_Lock)
            {
                if (m_NameToId.TryGetValue(behaviorName, out var id))
                {
                    return id;
                }

                id = m_IdToName.Count;
                m_NameToId[behaviorName] = id;
                m_IdToName.Add(behaviorName);
                return id;
            }
        }

        /// <summary>
        /// Gets the integer ID for a behavior name, or -1 if not registered.
        /// </summary>
        /// <param name="behaviorName">The behavior name string.</param>
        /// <returns>The integer ID, or -1 if not found.</returns>
        public int GetId(string behaviorName)
        {
            lock (m_Lock)
            {
                return m_NameToId.TryGetValue(behaviorName, out var id) ? id : -1;
            }
        }

        /// <summary>
        /// Gets the behavior name for an integer ID.
        /// </summary>
        /// <param name="id">The integer ID.</param>
        /// <returns>The behavior name string.</returns>
        public string GetName(int id)
        {
            lock (m_Lock)
            {
                if (id >= 0 && id < m_IdToName.Count)
                {
                    return m_IdToName[id];
                }
                return null;
            }
        }

        /// <summary>
        /// Gets the total number of registered behaviors.
        /// </summary>
        public int Count
        {
            get
            {
                lock (m_Lock)
                {
                    return m_IdToName.Count;
                }
            }
        }

        /// <summary>
        /// Clears all registered behaviors.
        /// </summary>
        public void Clear()
        {
            lock (m_Lock)
            {
                m_NameToId.Clear();
                m_IdToName.Clear();
            }
        }

        /// <summary>
        /// Gets all registered behavior names.
        /// </summary>
        public IEnumerable<string> GetAllNames()
        {
            lock (m_Lock)
            {
                return new List<string>(m_IdToName);
            }
        }
    }
}
#endif
