#if UNITY_EDITOR || UNITY_STANDALONE
using NUnit.Framework;
using Unity.MLAgents;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class BehaviorIdRegistryTests
    {
        private BehaviorIdRegistry m_Registry;

        [SetUp]
        public void SetUp()
        {
            m_Registry = new BehaviorIdRegistry();
        }

        [Test]
        public void GetOrCreateId_AssignsIncrementingIds()
        {
            var id0 = m_Registry.GetOrCreateId("behavior_a");
            var id1 = m_Registry.GetOrCreateId("behavior_b");
            var id2 = m_Registry.GetOrCreateId("behavior_c");

            Assert.AreEqual(0, id0);
            Assert.AreEqual(1, id1);
            Assert.AreEqual(2, id2);
        }

        [Test]
        public void GetOrCreateId_ReturnsSameIdForSameName()
        {
            var id1 = m_Registry.GetOrCreateId("test_behavior");
            var id2 = m_Registry.GetOrCreateId("test_behavior");

            Assert.AreEqual(id1, id2);
        }

        [Test]
        public void GetId_ReturnsMinusOneForUnregistered()
        {
            var id = m_Registry.GetId("nonexistent");
            Assert.AreEqual(-1, id);
        }

        [Test]
        public void GetId_ReturnsCorrectIdForRegistered()
        {
            m_Registry.GetOrCreateId("test_behavior");
            var id = m_Registry.GetId("test_behavior");
            Assert.AreEqual(0, id);
        }

        [Test]
        public void GetName_ReturnsCorrectName()
        {
            m_Registry.GetOrCreateId("first_behavior");
            m_Registry.GetOrCreateId("second_behavior");

            Assert.AreEqual("first_behavior", m_Registry.GetName(0));
            Assert.AreEqual("second_behavior", m_Registry.GetName(1));
        }

        [Test]
        public void GetName_ReturnsNullForInvalidId()
        {
            Assert.IsNull(m_Registry.GetName(-1));
            Assert.IsNull(m_Registry.GetName(100));
        }

        [Test]
        public void Count_ReturnsCorrectCount()
        {
            Assert.AreEqual(0, m_Registry.Count);

            m_Registry.GetOrCreateId("a");
            Assert.AreEqual(1, m_Registry.Count);

            m_Registry.GetOrCreateId("b");
            Assert.AreEqual(2, m_Registry.Count);

            // Same name shouldn't increase count
            m_Registry.GetOrCreateId("a");
            Assert.AreEqual(2, m_Registry.Count);
        }

        [Test]
        public void Clear_RemovesAllEntries()
        {
            m_Registry.GetOrCreateId("a");
            m_Registry.GetOrCreateId("b");
            Assert.AreEqual(2, m_Registry.Count);

            m_Registry.Clear();
            Assert.AreEqual(0, m_Registry.Count);
            Assert.AreEqual(-1, m_Registry.GetId("a"));
        }

        [Test]
        public void GetAllNames_ReturnsAllRegisteredNames()
        {
            m_Registry.GetOrCreateId("alpha");
            m_Registry.GetOrCreateId("beta");
            m_Registry.GetOrCreateId("gamma");

            var names = new System.Collections.Generic.List<string>(m_Registry.GetAllNames());

            Assert.AreEqual(3, names.Count);
            Assert.Contains("alpha", names);
            Assert.Contains("beta", names);
            Assert.Contains("gamma", names);
        }
    }
}
#endif
