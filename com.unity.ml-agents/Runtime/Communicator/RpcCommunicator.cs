#if UNITY_EDITOR || UNITY_STANDALONE
#define MLA_SUPPORTED_TRAINING_PLATFORM
#endif

#if MLA_SUPPORTED_TRAINING_PLATFORM
using Grpc.Core;
#if UNITY_EDITOR
using UnityEditor;
#endif
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.CommunicatorObjects;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.SideChannels;
using Google.Protobuf;

using Unity.MLAgents.Analytics;

namespace Unity.MLAgents
{
    /// Responsible for communication with External using gRPC.
    public class RpcCommunicator : ICommunicator
    {
        public event QuitCommandHandler QuitCommandReceived;
        public event ResetCommandHandler ResetCommandReceived;

        /// If true, the communication is active.
        bool m_IsOpen;

        List<string> m_BehaviorNames = new List<string>();
        bool m_NeedCommunicateThisStep;
        ObservationWriter m_ObservationWriter = new ObservationWriter();

        // Behavior ID registry for fast integer-based lookups
        BehaviorIdRegistry m_BehaviorIdRegistry = new BehaviorIdRegistry();

        // Use integer IDs internally for faster dictionary operations
        Dictionary<int, SensorShapeValidator> m_SensorShapeValidators = new Dictionary<int, SensorShapeValidator>();
        Dictionary<int, List<int>> m_OrderedAgentsRequestingDecisions = new Dictionary<int, List<int>>();

        /// The current UnityRLOutput to be sent when all the brains queried the communicator
        UnityRLOutputProto m_CurrentUnityRlOutput =
            new UnityRLOutputProto();

        // Use integer behavior IDs for inner dictionary lookup
        Dictionary<int, Dictionary<int, ActionBuffers>> m_LastActionsReceived =
            new Dictionary<int, Dictionary<int, ActionBuffers>>();

        // Brains that we have sent over the communicator with agents.
        HashSet<int> m_SentBrainKeys = new HashSet<int>();
        Dictionary<int, ActionSpec> m_UnsentBrainKeys = new Dictionary<int, ActionSpec>();

        /// The Unity to External client.
        UnityToExternalProto.UnityToExternalProtoClient m_Client;
        Channel m_Channel;

        /// <summary>
        /// Initializes a new instance of the RPCCommunicator class.
        /// </summary>
        protected RpcCommunicator()
        {
        }

        public static RpcCommunicator Create()
        {
#if MLA_SUPPORTED_TRAINING_PLATFORM
            return new RpcCommunicator();
#else
            return null;
#endif
        }

        #region Initialization

        internal static bool CheckCommunicationVersionsAreCompatible(
            string unityCommunicationVersion,
            string pythonApiVersion
        )
        {
            var unityVersion = new Version(unityCommunicationVersion);
            var pythonVersion = new Version(pythonApiVersion);
            if (unityVersion.Major == 0)
            {
                if (unityVersion.Major != pythonVersion.Major || unityVersion.Minor != pythonVersion.Minor)
                {
                    return false;
                }
            }
            else if (unityVersion.Major != pythonVersion.Major)
            {
                return false;
            }
            else if (unityVersion.Minor != pythonVersion.Minor)
            {
                // If a feature is used in Unity but not supported in the trainer,
                // we will warn at the point it's used. Don't warn here to avoid noise.
            }
            return true;
        }

        /// <summary>
        /// Sends the initialization parameters through the Communicator.
        /// Is used by the academy to send initialization parameters to the communicator.
        /// </summary>
        /// <returns>Whether the connection was successful.</returns>
        /// <param name="initParameters">The Unity Initialization Parameters to be sent.</param>
        /// <param name="initParametersOut">The External Initialization Parameters received.</param>
        public bool Initialize(CommunicatorInitParameters initParameters, out UnityRLInitParameters initParametersOut)
        {
#if MLA_SUPPORTED_TRAINING_PLATFORM
            var academyParameters = new UnityRLInitializationOutputProto
            {
                Name = initParameters.name,
                PackageVersion = initParameters.unityPackageVersion,
                CommunicationVersion = initParameters.unityCommunicationVersion,
                Capabilities = initParameters.CSharpCapabilities.ToProto()
            };

            UnityInputProto input;
            UnityInputProto initializationInput;
            try
            {
                initializationInput = Initialize(
                    initParameters.port,
                    new UnityOutputProto
                    {
                        RlInitializationOutput = academyParameters
                    },
                    out input
                );
            }
            catch (Exception ex)
            {
                if (ex is RpcException rpcException)
                {
                    switch (rpcException.Status.StatusCode)
                    {
                        case StatusCode.Unavailable:
                            // This is the common case where there's no trainer to connect to.
                            break;
                        case StatusCode.DeadlineExceeded:
                            // We don't currently set a deadline for connection, but likely will in the future.
                            break;
                        default:
                            Debug.Log($"Unexpected gRPC exception when trying to initialize communication: {rpcException}");
                            break;
                    }
                }
                else
                {
                    Debug.Log($"Unexpected exception when trying to initialize communication: {ex}");
                }
                initParametersOut = new UnityRLInitParameters();
                NotifyQuitAndShutDownChannel();
                return false;
            }

            var pythonPackageVersion = initializationInput.RlInitializationInput.PackageVersion;
            var pythonCommunicationVersion = initializationInput.RlInitializationInput.CommunicationVersion;
            TrainingAnalytics.SetTrainerInformation(pythonPackageVersion, pythonCommunicationVersion);

            var communicationIsCompatible = CheckCommunicationVersionsAreCompatible(
                initParameters.unityCommunicationVersion,
                pythonCommunicationVersion
            );

            // Initialization succeeded part-way. The most likely cause is a mismatch between the communicator
            // API strings, so log an explicit warning if that's the case.
            if (initializationInput != null && input == null)
            {
                if (!communicationIsCompatible)
                {
                    Debug.LogWarningFormat(
                        "Communication protocol between python ({0}) and Unity ({1}) have different " +
                        "versions which make them incompatible. Python library version: {2}.",
                        pythonCommunicationVersion, initParameters.unityCommunicationVersion,
                        pythonPackageVersion
                    );
                }
                else
                {
                    Debug.LogWarningFormat(
                        "Unknown communication error between Python. Python communication protocol: {0}, " +
                        "Python library version: {1}.",
                        pythonCommunicationVersion,
                        pythonPackageVersion
                    );
                }

                initParametersOut = new UnityRLInitParameters();
                return false;
            }

            UpdateEnvironmentWithInput(input.RlInput);
            initParametersOut = initializationInput.RlInitializationInput.ToUnityRLInitParameters();
            // Be sure to shut down the grpc channel when the application is quitting.
            Application.quitting += NotifyQuitAndShutDownChannel;
            return true;
#else
            initParametersOut = new UnityRLInitParameters();
            return false;
#endif
        }

        /// <summary>
        /// Adds the brain to the list of brains which will be sending information to External.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        public void SubscribeBrain(string brainKey, ActionSpec actionSpec)
        {
            if (m_BehaviorNames.Contains(brainKey))
            {
                return;
            }
            m_BehaviorNames.Add(brainKey);

            // Register behavior with integer ID
            var behaviorId = m_BehaviorIdRegistry.GetOrCreateId(brainKey);

            m_CurrentUnityRlOutput.AgentInfos.Add(
                brainKey,
                new UnityRLOutputProto.Types.ListAgentInfoProto()
            );

            CacheActionSpec(brainKey, actionSpec);
        }

        void UpdateEnvironmentWithInput(UnityRLInputProto rlInput)
        {
            SideChannelManager.ProcessSideChannelData(rlInput.SideChannel.ToArray());
            SendCommandEvent(rlInput.Command);
        }

        UnityInputProto Initialize(int port, UnityOutputProto unityOutput, out UnityInputProto unityInput)
        {
            m_IsOpen = true;
            m_Channel = new Channel($"localhost:{port}", ChannelCredentials.Insecure);

            m_Client = new UnityToExternalProto.UnityToExternalProtoClient(m_Channel);
            var result = m_Client.Exchange(WrapMessage(unityOutput, 200));
            var inputMessage = m_Client.Exchange(WrapMessage(null, 200));
            unityInput = inputMessage.UnityInput;
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#endif
            if (result.Header.Status != 200 || inputMessage.Header.Status != 200)
            {
                m_IsOpen = false;
                NotifyQuitAndShutDownChannel();
            }
            return result.UnityInput;
        }

        void NotifyQuitAndShutDownChannel()
        {
            QuitCommandReceived?.Invoke();
            try
            {
                m_Channel.ShutdownAsync().Wait();
            }
            catch (Exception)
            {
                // do nothing
            }
        }

        #endregion

        #region Destruction

        /// <summary>
        /// Close the communicator gracefully on both sides of the communication.
        /// </summary>
        public void Dispose()
        {
            if (!m_IsOpen)
            {
                return;
            }

            try
            {
                m_Client.Exchange(WrapMessage(null, 400));
                m_IsOpen = false;
            }
            catch
            {
                // ignored
            }
        }

        #endregion

        #region Sending Events

        void SendCommandEvent(CommandProto command)
        {
            switch (command)
            {
                case CommandProto.Quit:
                {
                    NotifyQuitAndShutDownChannel();
                    return;
                }
                case CommandProto.Reset:
                {
                    foreach (var brainName in m_OrderedAgentsRequestingDecisions.Keys)
                    {
                        m_OrderedAgentsRequestingDecisions[brainName].Clear();
                    }
                    ResetCommandReceived?.Invoke();
                    return;
                }
                default:
                {
                    return;
                }
            }
        }

        #endregion

        #region Sending and retreiving data

        public void DecideBatch()
        {
            if (!m_NeedCommunicateThisStep)
            {
                return;
            }
            m_NeedCommunicateThisStep = false;

            SendBatchedMessageHelper();
        }

        /// <summary>
        /// Sends the observations of one Agent.
        /// </summary>
        /// <param name="behaviorName">Batch Key.</param>
        /// <param name="info">Agent info.</param>
        /// <param name="sensors">Sensors that will produce the observations</param>
        public void PutObservations(string behaviorName, AgentInfo info, List<ISensor> sensors)
        {
            // Convert string to integer ID for faster internal lookups
            var behaviorId = m_BehaviorIdRegistry.GetOrCreateId(behaviorName);

#if DEBUG
            if (!m_SensorShapeValidators.TryGetValue(behaviorId, out var validator))
            {
                validator = new SensorShapeValidator();
                m_SensorShapeValidators[behaviorId] = validator;
            }
            validator.ValidateSensors(sensors);
#endif

            using (TimerStack.Instance.Scoped("AgentInfo.ToProto"))
            {
                var agentInfoProto = info.ToAgentInfoProto();

                using (TimerStack.Instance.Scoped("GenerateSensorData"))
                {
                    foreach (var sensor in sensors)
                    {
                        var obsProto = sensor.GetObservationProto(m_ObservationWriter);
                        agentInfoProto.Observations.Add(obsProto);
                    }
                }
                m_CurrentUnityRlOutput.AgentInfos[behaviorName].Value.Add(agentInfoProto);
            }

            m_NeedCommunicateThisStep = true;
            if (!m_OrderedAgentsRequestingDecisions.TryGetValue(behaviorId, out var orderedAgents))
            {
                orderedAgents = new List<int>();
                m_OrderedAgentsRequestingDecisions[behaviorId] = orderedAgents;
            }
            if (!info.done)
            {
                orderedAgents.Add(info.episodeId);
            }
            if (!m_LastActionsReceived.TryGetValue(behaviorId, out var behaviorActions))
            {
                behaviorActions = new Dictionary<int, ActionBuffers>();
                m_LastActionsReceived[behaviorId] = behaviorActions;
            }
            behaviorActions[info.episodeId] = ActionBuffers.Empty;
            if (info.done)
            {
                behaviorActions.Remove(info.episodeId);
            }
        }

        /// <summary>
        /// Helper method that sends the current UnityRLOutput, receives the next UnityInput and
        /// Applies the appropriate AgentAction to the agents.
        /// </summary>
        void SendBatchedMessageHelper()
        {
            var message = new UnityOutputProto
            {
                RlOutput = m_CurrentUnityRlOutput,
            };
            var tempUnityRlInitializationOutput = GetTempUnityRlInitializationOutput();
            if (tempUnityRlInitializationOutput != null)
            {
                message.RlInitializationOutput = tempUnityRlInitializationOutput;
            }

            byte[] messageAggregated = SideChannelManager.GetSideChannelMessage();
            message.RlOutput.SideChannel = ByteString.CopyFrom(messageAggregated);

            var input = Exchange(message);

            UpdateSentActionSpec(tempUnityRlInitializationOutput);

            foreach (var k in m_CurrentUnityRlOutput.AgentInfos.Keys)
            {
                m_CurrentUnityRlOutput.AgentInfos[k].Value.Clear();
            }

            var rlInput = input?.RlInput;

            if (rlInput?.AgentActions == null)
            {
                return;
            }

            UpdateEnvironmentWithInput(rlInput);

            foreach (var brainName in rlInput.AgentActions.Keys)
            {
                // Convert string to int for internal lookup
                var behaviorId = m_BehaviorIdRegistry.GetOrCreateId(brainName);

                if (!m_OrderedAgentsRequestingDecisions.TryGetValue(behaviorId, out var orderedAgents) || !orderedAgents.Any())
                {
                    continue;
                }

                if (!rlInput.AgentActions[brainName].Value.Any())
                {
                    continue;
                }

                var agentActions = rlInput.AgentActions[brainName].ToAgentActionList();
                var numAgents = orderedAgents.Count;
                for (var i = 0; i < numAgents; i++)
                {
                    var agentAction = agentActions[i];
                    var agentId = orderedAgents[i];
                    if (m_LastActionsReceived.TryGetValue(behaviorId, out var brainActions))
                    {
                        brainActions[agentId] = agentAction;
                    }
                }
            }
            foreach (var behaviorId in m_OrderedAgentsRequestingDecisions.Keys)
            {
                m_OrderedAgentsRequestingDecisions[behaviorId].Clear();
            }
        }

        public ActionBuffers GetActions(string behaviorName, int agentId)
        {
            // Convert string to int for fast lookup
            var behaviorId = m_BehaviorIdRegistry.GetId(behaviorName);
            if (behaviorId < 0)
            {
                return ActionBuffers.Empty;
            }

            if (m_LastActionsReceived.TryGetValue(behaviorId, out var agentActions))
            {
                if (agentActions.TryGetValue(agentId, out var action))
                {
                    return action;
                }
            }
            return ActionBuffers.Empty;
        }

        /// <summary>
        /// Send a UnityOutput and receives a UnityInput.
        /// </summary>
        /// <returns>The next UnityInput.</returns>
        /// <param name="unityOutput">The UnityOutput to be sent.</param>
        UnityInputProto Exchange(UnityOutputProto unityOutput)
        {
            if (!m_IsOpen)
            {
                return null;
            }

            try
            {
                var message = m_Client.Exchange(WrapMessage(unityOutput, 200));
                if (message.Header.Status == 200)
                {
                    return message.UnityInput;
                }

                m_IsOpen = false;
                // Not sure if the quit command is actually sent when a
                // non 200 message is received.  Notify that we are indeed
                // quitting.
                NotifyQuitAndShutDownChannel();
                return message.UnityInput;
            }
            catch (Exception ex)
            {
                if (ex is RpcException rpcException)
                {
                    // Log more verbose errors if they're something the user can possibly do something about.
                    switch (rpcException.Status.StatusCode)
                    {
                        case StatusCode.Unavailable:
                            // This can happen when python disconnects. Ignore it to avoid noisy logs.
                            break;
                        case StatusCode.ResourceExhausted:
                            // This happens is the message body is too large. There's no way to
                            // gracefully handle this, but at least we can show the message and the
                            // user can try to reduce the number of agents or observation sizes.
                            Debug.LogError($"GRPC Exception: {rpcException.Message}. Disconnecting from trainer.");
                            break;
                        default:
                            // Other unknown errors. Log at INFO level.
                            Debug.Log($"GRPC Exception: {rpcException.Message}. Disconnecting from trainer.");
                            break;
                    }
                }
                else
                {
                    // Fall-through for other error types
                    Debug.LogError($"Communication Exception: {ex.Message}. Disconnecting from trainer.");
                }

                m_IsOpen = false;
                NotifyQuitAndShutDownChannel();
                return null;
            }
        }

        /// <summary>
        /// Wraps the UnityOutput into a message with the appropriate status.
        /// </summary>
        /// <returns>The UnityMessage corresponding.</returns>
        /// <param name="content">The UnityOutput to be wrapped.</param>
        /// <param name="status">The status of the message.</param>
        static UnityMessageProto WrapMessage(UnityOutputProto content, int status)
        {
            return new UnityMessageProto
            {
                Header = new HeaderProto { Status = status },
                UnityOutput = content
            };
        }

        void CacheActionSpec(string behaviorName, ActionSpec actionSpec)
        {
            var behaviorId = m_BehaviorIdRegistry.GetOrCreateId(behaviorName);

            if (m_SentBrainKeys.Contains(behaviorId))
            {
                return;
            }

            // TODO We should check that if m_unsentBrainKeys has brainKey, it equals actionSpec
            m_UnsentBrainKeys[behaviorId] = actionSpec;
        }

        UnityRLInitializationOutputProto GetTempUnityRlInitializationOutput()
        {
            UnityRLInitializationOutputProto output = null;
            foreach (var kvp in m_UnsentBrainKeys)
            {
                var behaviorId = kvp.Key;
                var actionSpec = kvp.Value;

                // Convert back to string for protobuf message
                var behaviorName = m_BehaviorIdRegistry.GetName(behaviorId);
                if (behaviorName == null)
                    continue;

                if (m_CurrentUnityRlOutput.AgentInfos.TryGetValue(behaviorName, out var agentInfoList))
                {
                    if (agentInfoList.CalculateSize() > 0)
                    {
                        // Only send the actionSpec if there is a non empty list of
                        // AgentInfos ready to be sent.
                        // This is to ensure that The Python side will always have a first
                        // observation when receiving the ActionSpec
                        if (output == null)
                        {
                            output = new UnityRLInitializationOutputProto();
                        }

                        output.BrainParameters.Add(actionSpec.ToBrainParametersProto(behaviorName, true));
                    }
                }
            }

            return output;
        }

        void UpdateSentActionSpec(UnityRLInitializationOutputProto output)
        {
            if (output == null)
            {
                return;
            }

            foreach (var brainProto in output.BrainParameters)
            {
                var behaviorId = m_BehaviorIdRegistry.GetOrCreateId(brainProto.BrainName);
                m_SentBrainKeys.Add(behaviorId);
                m_UnsentBrainKeys.Remove(behaviorId);
            }
        }

        #endregion

#if UNITY_EDITOR
        /// <summary>
        /// When the editor exits, the communicator must be closed
        /// </summary>
        /// <param name="state">State.</param>
        void HandleOnPlayModeChanged(PlayModeStateChange state)
        {
            // This method is run whenever the playmode state is changed.
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                Dispose();
            }
        }

#endif
    }
}
#endif // UNITY_EDITOR || UNITY_STANDALONE
