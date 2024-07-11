from subprocess import Popen, PIPE


# # This command will actually start an AWS instance:
# process = Popen(['aws', 'ec2', 'run-instances', '--instance-type', 't2.micro',
#                  '--image-id', 'ami-0ccabb5f82d4c9af5'], stdout=PIPE, stderr=PIPE)

# Use this instead to make sure AWS CLI is working:
process = Popen(['aws', '--version'], stdout=PIPE, stderr=PIPE)

stdout, stderr = process.communicate()

print(f'stdout: {stdout.decode("utf-8")}')
print(f'stderr: {stderr.decode("utf-8")}')
