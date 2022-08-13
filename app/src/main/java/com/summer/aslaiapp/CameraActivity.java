package com.summer.aslaiapp;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

/** Main {@code Activity} class for the Camera app. */
public class CameraActivity extends AppCompatActivity {
    Camera2BasicFragment camera;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        camera = Camera2BasicFragment.newInstance();
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.container, camera)
                .commit();
        findViewById(R.id.camera_button).setOnClickListener(view1 -> {
            camera.switchCamera();
        });
    }
}
