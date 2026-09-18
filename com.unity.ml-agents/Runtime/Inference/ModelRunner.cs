using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine.Profiling;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Inference
{
    internal struct AgentInfoSensorsPair
    {
        public AgentInfo agentInfo;
        public List<ISensor> sensors;
    }

    internal class ModelRunner
    {
        // Pre-allocated capacity for better performance with many agents
        const int k_DefaultBatchCapacity = 512;

        List<AgentInfoSensorsPair> m_Infos;
        Dictionary<int, ActionBuffers> m_LastActionsReceived;
        List<int> m_OrderedAgentsRequestingDecisions;

        // Array-based storage for batch processing (avoids dictionary lookups in hot path)
        ActionBuffers[] m_BatchedActions;
        int m_CurrentBatchSize;

        TensorGenerator m_TensorGenerator;
        TensorApplier m_TensorApplier;

        ModelAsset m_Model;
        string m_ModelName;
        InferenceDevice m_InferenceDevice;
        Worker m_Engine;
        bool m_DeterministicInference;
        string[] m_OutputNames;
        IReadOnlyList<TensorProxy> m_InferenceInputs;
        List<TensorProxy> m_InferenceOutputs;
        Dictionary<string, Tensor> m_InputsByName;
        Dictionary<int, List<float>> m_Memories = new Dictionary<int, List<float>>();

        SensorShapeValidator m_SensorShapeValidator = new SensorShapeValidator();

        bool m_ObservationsInitialized;

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model"> The Sentis model to load </param>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        /// <param name="inferenceDevice"> Inference execution device. CPU is the fastest
        /// option for most of ML Agents models. </param>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial objects used when running inference.</param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public ModelRunner(
            ModelAsset model,
            ActionSpec actionSpec,
            InferenceDevice inferenceDevice,
            int seed = 0,
            bool deterministicInference = false)
        {
            // Initialize collections with capacity hints for better performance
            m_Infos = new List<AgentInfoSensorsPair>(k_DefaultBatchCapacity);
            m_LastActionsReceived = new Dictionary<int, ActionBuffers>(k_DefaultBatchCapacity);
            m_OrderedAgentsRequestingDecisions = new List<int>(k_DefaultBatchCapacity);
            m_BatchedActions = new ActionBuffers[k_DefaultBatchCapacity];
            m_CurrentBatchSize = 0;

            Model sentisModel;
            SentisModelInfo sentisModelInfo;
            m_Model = model;
            m_ModelName = model?.name;
            m_InferenceDevice = inferenceDevice;
            m_DeterministicInference = deterministicInference;
            if (model != null)
            {
#if SENTIS_VERBOSE
                m_Verbose = true;
#endif

                // TODO check w/Alex about verbosity level
                // D.logEnabled = m_Verbose;

                sentisModel = ModelLoader.Load(model);
                sentisModelInfo = new SentisModelInfo(sentisModel, deterministicInference);

                var failedCheck = SentisModelParamLoader.CheckModelVersion(
                    sentisModelInfo
                );
                if (failedCheck != null)
                {
                    if (failedCheck.CheckType == SentisModelParamLoader.FailedCheck.CheckTypeEnum.Error)
                    {
                        throw new UnityAgentsException(failedCheck.Message);
                    }
                }

                BackendType executionDevice;
                // WorkerFactory.Type executionDevice;
                switch (inferenceDevice)
                {
                    case InferenceDevice.ComputeShader:
                        executionDevice = BackendType.GPUCompute;
                        break;
                    case InferenceDevice.PixelShader:
                        executionDevice = BackendType.GPUPixel;
                        break;
                    case InferenceDevice.Burst:
                        executionDevice = BackendType.CPU;
                        break;
                    case InferenceDevice.Default: // fallthrough
                    default:
                        executionDevice = BackendType.CPU;
                        break;
                }
                m_Engine = new Worker(sentisModel, executionDevice);
            }
            else
            {
                sentisModel = null;
                sentisModelInfo = null;
                m_Engine = null;
            }

            if (sentisModelInfo != null)
            {
                m_InferenceInputs = sentisModelInfo.GetInputTensors();
                m_OutputNames = sentisModelInfo.OutputNames;
            }

            m_TensorGenerator = new TensorGenerator(
                seed, m_Memories, sentisModel, m_DeterministicInference);
            m_TensorApplier = new TensorApplier(
                actionSpec, seed, m_Memories, sentisModel, m_DeterministicInference);
            m_InputsByName = new Dictionary<string, Tensor>();
            m_InferenceOutputs = new List<TensorProxy>();
            sentisModelInfo?.Dispose();
        }

        public InferenceDevice InferenceDevice
        {
            get { return m_InferenceDevice; }
        }

        public ModelAsset Model
        {
            get { return m_Model; }
        }

        void PrepareSentisInputs(IReadOnlyList<TensorProxy> infInputs)
        {
            m_InputsByName.Clear();
            for (var i = 0; i < infInputs.Count; i++)
            {
                var inp = infInputs[i];
                m_InputsByName[inp.name] = inp.data;
            }
        }

        public void Dispose()
        {
            if (m_Engine != null)
                m_Engine.Dispose();
            foreach (var (name, tensor) in m_InputsByName)
            {
                tensor.Dispose();
            }
        }

        void FetchSentisOutputs(string[] names)
        {
            // Dispose explicitly, on the main thread, right here — instead of leaving these
            // TensorProxy wrappers to the GC finalizer, which runs on a background thread and
            // isn't safe for these native-backed tensors (each disposal is a no-op on the
            // underlying data since OwnsData is false for output wrappers, but the TensorProxy
            // itself still needs Dispose() to run its cleanup and suppress its finalizer).
            foreach (var proxy in m_InferenceOutputs)
            {
                proxy.Dispose();
            }
            m_InferenceOutputs.Clear();

            foreach (var n in names)
            {
                var output = m_Engine.PeekOutput(n);
                m_InferenceOutputs.Add(TensorUtils.TensorProxyFromSentis(output, n));
            }
        }

        public void PutObservations(AgentInfo info, List<ISensor> sensors)
        {
#if DEBUG
            m_SensorShapeValidator.ValidateSensors(sensors);
#endif
            m_Infos.Add(new AgentInfoSensorsPair
            {
                agentInfo = info,
                sensors = sensors
            });

            // We add the episodeId to this list to maintain the order in which the decisions were requested
            m_OrderedAgentsRequestingDecisions.Add(info.episodeId);

            if (!m_LastActionsReceived.ContainsKey(info.episodeId))
            {
                m_LastActionsReceived[info.episodeId] = ActionBuffers.Empty;
            }
            if (info.done)
            {
                // If the agent is done, we remove the key from the last action dictionary since no action
                // should be taken.
                m_LastActionsReceived.Remove(info.episodeId);
            }
        }

        public void DecideBatch()
        {
            var currentBatchSize = m_Infos.Count;
            if (currentBatchSize == 0)
            {
                return;
            }

            // Ensure batch arrays are large enough
            EnsureBatchCapacity(currentBatchSize);
            m_CurrentBatchSize = currentBatchSize;

            if (!m_ObservationsInitialized)
            {
                // Just grab the first agent in the collection (any will suffice, really).
                // We check for an empty Collection above, so this will always return successfully.
                var firstInfo = m_Infos[0];
                m_TensorGenerator.InitializeObservations(firstInfo.sensors);
                m_ObservationsInitialized = true;
            }

            Profiler.BeginSample("ModelRunner.DecideAction");
            Profiler.BeginSample(m_ModelName);

            Profiler.BeginSample("GenerateTensors");
            // Prepare the input tensors to be feed into the engine
            m_TensorGenerator.GenerateTensors(m_InferenceInputs, currentBatchSize, m_Infos);
            Profiler.EndSample();

            Profiler.BeginSample("PrepareSentisInputs");
            PrepareSentisInputs(m_InferenceInputs);
            Profiler.EndSample();

            // Execute the Model
            Profiler.BeginSample("ExecuteGraph");
            foreach (var kv in m_InputsByName)
            {
                m_Engine.SetInput(kv.Key, kv.Value);
            }
            m_Engine.Schedule();
            Profiler.EndSample();

            Profiler.BeginSample("FetchSentisOutputs");
            FetchSentisOutputs(m_OutputNames);
            Profiler.EndSample();

            Profiler.BeginSample("ApplyTensors");
            // Update the outputs using optimized batch arrays
            m_TensorApplier.ApplyTensors(m_InferenceOutputs, m_OrderedAgentsRequestingDecisions, m_LastActionsReceived);
            Profiler.EndSample();

            // Sync batch results back to dictionary for GetAction lookups
            Profiler.BeginSample("SyncBatchResults");
            SyncBatchResultsToDictionary();
            Profiler.EndSample();

            Profiler.EndSample(); // end name
            Profiler.EndSample(); // end ModelRunner.DecideAction

            m_Infos.Clear();
            m_OrderedAgentsRequestingDecisions.Clear();
        }

        void EnsureBatchCapacity(int requiredCapacity)
        {
            if (m_BatchedActions.Length < requiredCapacity)
            {
                // Grow by 2x to amortize allocations
                var newCapacity = System.Math.Max(requiredCapacity, m_BatchedActions.Length * 2);
                System.Array.Resize(ref m_BatchedActions, newCapacity);
            }
        }

        void SyncBatchResultsToDictionary()
        {
            // After ApplyTensors updates the dictionary, we can optionally cache in arrays
            // for faster subsequent GetAction calls within the same step
            for (var i = 0; i < m_CurrentBatchSize && i < m_OrderedAgentsRequestingDecisions.Count; i++)
            {
                var agentId = m_OrderedAgentsRequestingDecisions[i];
                if (m_LastActionsReceived.TryGetValue(agentId, out var action))
                {
                    m_BatchedActions[i] = action;
                }
            }
        }

        public bool HasModel(ModelAsset other, InferenceDevice otherInferenceDevice)
        {
            return m_Model == other && m_InferenceDevice == otherInferenceDevice;
        }

        public ActionBuffers GetAction(int agentId)
        {
            if (m_LastActionsReceived.TryGetValue(agentId, out var action))
            {
                return action;
            }
            return ActionBuffers.Empty;
        }
    }
}
