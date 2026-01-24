// Automatically spawns additional training areas at runtime to increase parallelism
// Attach this script to any GameObject in your scene, or it will auto-create itself
// Configure the number of copies in the Inspector or via the static defaults

using System.Collections.Generic;
using UnityEngine;

namespace MLAgentsExamples
{
    /// <summary>
    /// Automatically duplicates training areas at runtime to increase training parallelism.
    /// Works with any ML-Agents training scene (Walker, 3DBall, Crawler, etc.)
    /// 
    /// Usage:
    /// 1. Add this script to any GameObject in your scene, OR
    /// 2. It will auto-create itself when Play is pressed
    /// 3. Set the number of total areas you want in the Inspector
    /// 
    /// The script will find existing training areas and duplicate them in a grid pattern.
    /// </summary>
    public class AutoSpawnTrainingAreas : MonoBehaviour
    {
        [Header("Spawning Settings")]
        [Tooltip("Total number of training areas to have (including originals)")]
        [Range(1, 100)]
        public int totalAreas = 48;
        
        [Tooltip("Spacing between training areas")]
        public float spacing = 50f;
        
        [Tooltip("Number of areas per row in the grid")]
        public int areasPerRow = 6;
        
        [Header("Auto-Detection")]
        [Tooltip("Keywords to search for in GameObject names to find training areas")]
        public string[] areaKeywords = { "Area", "Training", "Walker", "Crawler", "3DBall", "Hallway", "Agent" };
        
        [Header("Debug")]
        [Tooltip("Show debug messages in console")]
        public bool showDebugMessages = true;
        
        private static AutoSpawnTrainingAreas _instance;
        private List<GameObject> _spawnedAreas = new List<GameObject>();
        private bool _hasSpawned = false;
        private bool _isAutoCreated = false;

        // Static configuration for auto-creation mode
        private static int _defaultTotalAreas = 48;
        private static float _defaultSpacing = 50f;
        
        /// <summary>
        /// Set the default number of areas before Play mode starts
        /// </summary>
        public static void SetDefaults(int totalAreas, float spacing = 50f)
        {
            _defaultTotalAreas = totalAreas;
            _defaultSpacing = spacing;
        }

        // DISABLED: Runtime instantiation causes observation count mismatch with ML-Agents
        // The cloned agents have duplicated sensors, causing "More observations made than vector observation size" errors
        // Instead, duplicate training areas manually in the Unity Editor (Ctrl+D) and position them
        // [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate_DISABLED()
        {
            // Only auto-create if no instance exists
            if (_instance != null) return;
            
            // Check if one already exists in the scene
            var existing = FindAnyObjectByType<AutoSpawnTrainingAreas>();
            if (existing != null)
            {
                _instance = existing;
                return;
            }
            
            var go = new GameObject("AutoSpawnTrainingAreas");
            var spawner = go.AddComponent<AutoSpawnTrainingAreas>();
            spawner.totalAreas = _defaultTotalAreas;
            spawner.spacing = _defaultSpacing;
            spawner._isAutoCreated = true;
            _instance = spawner;
            DontDestroyOnLoad(go);
        }

        void Awake()
        {
            // Register this instance if not already set
            if (_instance == null)
                _instance = this;
            else if (_instance != this && _isAutoCreated)
            {
                // Another instance exists and we're the auto-created one, destroy ourselves
                Destroy(gameObject);
                return;
            }
        }

        void Start()
        {
            if (_hasSpawned) return;
            
            SpawnAreas();
            _hasSpawned = true;
        }

        void OnDestroy()
        {
            // Clean up spawned areas
            foreach (var area in _spawnedAreas)
            {
                if (area != null)
                    Destroy(area);
            }
            _spawnedAreas.Clear();
            
            if (_instance == this)
                _instance = null;
        }

        /// <summary>
        /// Find and duplicate training areas
        /// </summary>
        public void SpawnAreas()
        {
            // Find existing training areas
            var existingAreas = FindTrainingAreas();
            
            if (existingAreas.Count == 0)
            {
                if (showDebugMessages)
                    Debug.LogWarning("[AutoSpawn] No training areas found to duplicate!");
                return;
            }

            int currentCount = existingAreas.Count;
            int toSpawn = totalAreas - currentCount;
            
            if (toSpawn <= 0)
            {
                if (showDebugMessages)
                    Debug.Log($"[AutoSpawn] Already have {currentCount} areas, target is {totalAreas}. No spawning needed.");
                return;
            }

            if (showDebugMessages)
                Debug.Log($"<color=cyan>[AutoSpawn] Found {currentCount} training areas. Spawning {toSpawn} more to reach {totalAreas} total...</color>");

            // Calculate the bounding box of existing areas to find good spawn positions
            Vector3 center = Vector3.zero;
            float maxExtent = 0f;
            
            foreach (var area in existingAreas)
            {
                center += area.transform.position;
                var bounds = GetBounds(area);
                maxExtent = Mathf.Max(maxExtent, bounds.extents.magnitude);
            }
            center /= existingAreas.Count;

            // Use the first area as the template for duplication
            GameObject template = existingAreas[0];
            
            // Calculate actual spacing (at least as big as the area bounds)
            float actualSpacing = Mathf.Max(spacing, maxExtent * 2.5f);

            // Spawn new areas in a grid pattern, offset from existing ones
            int spawned = 0;
            int row = 0;
            int col = existingAreas.Count; // Start from where existing ones end
            
            // Find max Z of existing areas to start new rows after them
            float maxZ = float.MinValue;
            float minX = float.MaxValue;
            foreach (var area in existingAreas)
            {
                maxZ = Mathf.Max(maxZ, area.transform.position.z);
                minX = Mathf.Min(minX, area.transform.position.x);
            }

            float startX = minX;
            float startZ = maxZ + actualSpacing;

            for (int i = 0; i < toSpawn; i++)
            {
                // Calculate grid position
                row = i / areasPerRow;
                col = i % areasPerRow;
                
                Vector3 newPosition = new Vector3(
                    startX + col * actualSpacing,
                    template.transform.position.y,
                    startZ + row * actualSpacing
                );

                // Instantiate the new area
                GameObject newArea = Instantiate(template, newPosition, template.transform.rotation);
                newArea.name = $"{template.name} (Spawned {currentCount + i + 1})";
                
                _spawnedAreas.Add(newArea);
                spawned++;
            }

            if (showDebugMessages)
                Debug.Log($"<color=green>[AutoSpawn] Successfully spawned {spawned} training areas! Total: {currentCount + spawned}</color>");
        }

        /// <summary>
        /// Find all training areas in the scene
        /// </summary>
        private List<GameObject> FindTrainingAreas()
        {
            var areas = new List<GameObject>();
            var allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);

            foreach (var obj in allObjects)
            {
                // Skip if it's a child of another object we might consider an area
                // (we want top-level areas only)
                if (obj.transform.parent != null)
                {
                    bool parentIsArea = false;
                    foreach (var keyword in areaKeywords)
                    {
                        if (obj.transform.parent.name.Contains(keyword))
                        {
                            parentIsArea = true;
                            break;
                        }
                    }
                    if (parentIsArea) continue;
                }

                // Check if name contains any of our keywords
                foreach (var keyword in areaKeywords)
                {
                    if (obj.name.Contains(keyword) && !obj.name.Contains("Spawned"))
                    {
                        // Additional check: must have some children (training areas usually have agents, ground, etc.)
                        if (obj.transform.childCount > 0)
                        {
                            areas.Add(obj);
                            break;
                        }
                    }
                }
            }

            return areas;
        }

        /// <summary>
        /// Get the bounds of a GameObject including all its children
        /// </summary>
        private Bounds GetBounds(GameObject obj)
        {
            var renderers = obj.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0)
                return new Bounds(obj.transform.position, Vector3.one * 10f);

            Bounds bounds = renderers[0].bounds;
            foreach (var renderer in renderers)
            {
                bounds.Encapsulate(renderer.bounds);
            }
            return bounds;
        }

        /// <summary>
        /// Remove all spawned areas (called automatically on destroy, but can be called manually)
        /// </summary>
        public void ClearSpawnedAreas()
        {
            foreach (var area in _spawnedAreas)
            {
                if (area != null)
                    DestroyImmediate(area);
            }
            _spawnedAreas.Clear();
            _hasSpawned = false;
            
            if (showDebugMessages)
                Debug.Log("[AutoSpawn] Cleared all spawned areas.");
        }
    }
}
