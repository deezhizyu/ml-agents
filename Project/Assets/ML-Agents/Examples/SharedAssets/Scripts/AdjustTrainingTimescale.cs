//This script lets you change time scale during training. It is not a required script for this demo to function
// Press 1-9 to set time scale, 0 to double current time scale

using UnityEngine;
using UnityEngine.InputSystem;

namespace MLAgentsExamples
{
    public class AdjustTrainingTimescale : MonoBehaviour
    {
        // Update is called once per frame
        void Update()
        {
            var keyboard = Keyboard.current;
            if (keyboard == null) return;

            if (keyboard.digit1Key.wasPressedThisFrame)
            {
                Time.timeScale = 1f;
                Debug.Log("Time scale set to 1x");
            }
            if (keyboard.digit2Key.wasPressedThisFrame)
            {
                Time.timeScale = 2f;
                Debug.Log("Time scale set to 2x");
            }
            if (keyboard.digit3Key.wasPressedThisFrame)
            {
                Time.timeScale = 3f;
                Debug.Log("Time scale set to 3x");
            }
            if (keyboard.digit4Key.wasPressedThisFrame)
            {
                Time.timeScale = 4f;
                Debug.Log("Time scale set to 4x");
            }
            if (keyboard.digit5Key.wasPressedThisFrame)
            {
                Time.timeScale = 5f;
                Debug.Log("Time scale set to 5x");
            }
            if (keyboard.digit6Key.wasPressedThisFrame)
            {
                Time.timeScale = 6f;
                Debug.Log("Time scale set to 6x");
            }
            if (keyboard.digit7Key.wasPressedThisFrame)
            {
                Time.timeScale = 7f;
                Debug.Log("Time scale set to 7x");
            }
            if (keyboard.digit8Key.wasPressedThisFrame)
            {
                Time.timeScale = 8f;
                Debug.Log("Time scale set to 8x");
            }
            if (keyboard.digit9Key.wasPressedThisFrame)
            {
                Time.timeScale = 9f;
                Debug.Log("Time scale set to 9x");
            }
            if (keyboard.digit0Key.wasPressedThisFrame)
            {
                Time.timeScale *= 2f;
                Debug.Log($"Time scale doubled to {Time.timeScale}x");
            }
        }
    }
}
