# Visu3d - 3d geometry for humans

3d is hard. visu3d makes it simple by:

*   Providing "semantic" primitives (Pose, Camera, Transformation...), so user
    can express intent, instead of math.

    ```python
    h, w = (256, 1024)
    cam_spec = v3d.PinholeCamera.from_focal(
        resolution=(h, w),
        focal_in_px=35.,
    )
    cam_pose = v3d.Pose.from_look_at(
        from_=[5, 5, 5],
        to=[0, 0, 0],  # Camera look at the scene center
    )
    cam = v3d.Camera(spec=cam_spec, pose=cam_pose)

    # Rays in world coordinates
    rays = cam.rays()  # -> Pose[h w]
    rays = rays.normalize()
    assert rays.shape == (h, w)

    # Ray origins and direction match the camera pose
    mean_ray = rays.mean()
    assert np.allclose(mean_ray.t, cam_pose.t)
    assert np.allclose(mean_ray.d, cam_pose.d)

    # Project rays back into the camera frame
    # TODO(epot)
    ```

*   Easy primitives manipulations: all primitives support batching though
    numpy-like API (indexing, slice,...)

    ```python
    assert rays.shape == (h, w)
    assert rays.t.shape == (h, w, 3)

    top_left_ray = rays[0, 0]
    mean_ray = rays.mean()
    ```

*   Native `einops` support:

    ```python
    rays = rays.reshape('h w -> (h w)')
    ```

*   Visualization as a first class citizen:

    ```python
    rays.fig.show()  # Visualize the rays
    # Display on the same figure the camera, average ray and
    # camera destination.
    v3d.make_fig([cam, rays.mean() * 3., cam.pose.td])
    ```

*   Same API supports both `np`, Jax and TF. Types are auto-inferred from
    inputs.

The best way to get started is to try the [colab](intro.ipynb).
